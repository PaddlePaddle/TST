import torch
import torch.fx as fx
from typing import Dict, List, Set, Union, Any, Tuple, Optional, Generator


# Viba type-aliases
class Uninitiated:
    pass


MatchResult = Dict[fx.Node, fx.Node]
FuzzyMatchResult = Dict[fx.Node, Union[Set[fx.Node], Uninitiated]]


def is_struct_match_for_sole_output_node(
    env: MatchResult, pattern: fx.Node, target: fx.Node
) -> Tuple[bool, MatchResult]:
    # Check env for consistency
    if pattern in env:
        return env[pattern] == target, env
    # match[$placeholder_pattern]
    if pattern.op == "placeholder":
        return True, {**env, pattern: target}
    # match[$compute_node_pattern]
    if pattern.op != target.op or pattern.target != target.target:
        return False, env
    if len(pattern.args) != len(target.args):
        return False, env
    # Reconcile argument streams
    curr_env = {**env, pattern: target}
    for p_arg, t_arg in zip(pattern.args, target.args):
        if isinstance(p_arg, fx.Node):
            if not isinstance(t_arg, fx.Node):
                return False, env
            ok, curr_env = is_struct_match_for_sole_output_node(curr_env, p_arg, t_arg)
            if not ok:
                return False, env
        elif p_arg != t_arg:
            return False, env
    return True, curr_env


def match_remainder_struct(
    existed_match: MatchResult,
    remainder_nodes: List[fx.Node],
    pattern: fx.GraphModule,
    target: fx.GraphModule,
) -> FuzzyMatchResult:
    # $merged_initial
    fuzzy_env: FuzzyMatchResult = {p: {t} for p, t in existed_match.items()}
    for p in remainder_nodes:
        if p not in fuzzy_env:
            fuzzy_env[p] = Uninitiated()

    all_compute = [n for n in pattern.graph.nodes if n.op != "output"]

    # $reach_fixed_point
    changed = True
    while changed:
        changed = False
        for p in all_compute:
            cands = (
                {t for t in target.graph.nodes if t.op == p.op and t.target == p.target}
                if p.op != "placeholder"
                else set(target.graph.nodes)
            )
            if not isinstance(fuzzy_env[p], Uninitiated):
                cands &= fuzzy_env[p]

            filtered = set()
            for cand in cands:
                if len(cand.args) != len(p.args):
                    continue
                # $defuzzy_by_inputs
                m_ok = True
                for i, p_a in enumerate(p.args):
                    if isinstance(p_a, fx.Node):
                        p_ds = fuzzy_env.get(p_a)
                        if (
                            not isinstance(p_ds, Uninitiated)
                            and cand.args[i] not in p_ds
                        ):
                            m_ok = False
                            break
                    elif p_a != cand.args[i]:
                        m_ok = False
                        break
                # $defuzzy_by_outputs
                if m_ok:
                    for p_u in p.users:
                        if p_u.op == "output":
                            continue
                        p_u_cs = fuzzy_env.get(p_u)
                        if not isinstance(p_u_cs, Uninitiated) and not any(
                            t_u in p_u_cs for t_u in cand.users
                        ):
                            m_ok = False
                            break
                if m_ok:
                    filtered.add(cand)

            if filtered != fuzzy_env[p]:
                fuzzy_env[p] = filtered
                changed = True
                if not filtered:
                    return {}
    return fuzzy_env


class SoleOutputPatternStructMatcher:
    def __init__(self, pattern_gm: fx.GraphModule):
        self.pattern_gm = pattern_gm
        self.root = next(n for n in pattern_gm.graph.nodes if n.op == "output").args[0]

    def __call__(self, target_gm: fx.GraphModule) -> List[MatchResult]:
        res = []
        # Support anchoring on sub-node if root is a list/tuple
        actual_root = (
            self.root[0] if isinstance(self.root, (list, tuple)) else self.root
        )
        for node in target_gm.graph.nodes:
            ok, match = is_struct_match_for_sole_output_node({}, actual_root, node)
            if ok:
                res.append(match)
        return res


class OutputTuplePatternStructMatcher:
    def __init__(self, pattern_gm: fx.GraphModule):
        self.pattern_gm = pattern_gm
        # $biggest_sole_sink_dag_in_pattern_dag via depth
        depths = {}

        def get_d(n):
            if n in depths:
                return depths[n]
            if n.op == "placeholder":
                return 0
            d = 1 + max([get_d(a) for a in n.args if isinstance(a, fx.Node)] + [0])
            depths[n] = d
            return d

        all_p = [n for n in pattern_gm.graph.nodes if n.op != "output"]
        self.anchor = max(all_p, key=get_d)
        self.remainder = all_p

    def __call__(self, target_gm: fx.GraphModule) -> List[MatchResult]:
        matches, final, seen = [], [], set()
        for node in target_gm.graph.nodes:
            ok, m = is_struct_match_for_sole_output_node({}, self.anchor, node)
            if ok:
                matches.append(m)
        for m in matches:
            fuzzy = match_remainder_struct(
                m, self.remainder, self.pattern_gm, target_gm
            )
            if fuzzy and all(len(v) == 1 for v in fuzzy.values() if isinstance(v, set)):
                res = {p: list(ts)[0] for p, ts in fuzzy.items() if isinstance(ts, set)}
                sig = tuple(sorted(n.name for n in res.values()))
                if sig not in seen:
                    final.append(res)
                    seen.add(sig)
        return final


class StructMatcher:
    def __init__(self, pattern_gm: fx.GraphModule):
        self.pattern_gm = pattern_gm
        out = next(n for n in pattern_gm.graph.nodes if n.op == "output").args[0]
        self.delegate = (
            OutputTuplePatternStructMatcher(pattern_gm)
            if isinstance(out, (list, tuple))
            else SoleOutputPatternStructMatcher(pattern_gm)
        )

    def __call__(self, target_gm: fx.GraphModule) -> List[MatchResult]:
        return self.delegate(target_gm)


def test_main():
    # 1. SoleOutput: Simple add
    p1 = fx.symbolic_trace(lambda x: x + 1)
    assert len(StructMatcher(p1)(p1)) == 1
    # 2. SoleOutput: Mismatch
    assert len(StructMatcher(p1)(fx.symbolic_trace(lambda x: x * 1))) == 0
    # 3. SoleOutput: Multiple matches in chain
    t3 = fx.symbolic_trace(lambda x: x + 1 + 1)
    assert len(StructMatcher(p1)(t3)) == 2
    # 4. SoleOutput: Diamond structure
    p4 = fx.symbolic_trace(lambda x: (x * 2) + (x * 2))
    assert len(StructMatcher(p4)(p4)) == 1
    # 5. SoleOutput: Permutation of inputs
    p5 = fx.symbolic_trace(lambda x, y: x - y)
    t5 = fx.symbolic_trace(lambda a, b: b - a)
    assert len(StructMatcher(p5)(t5)) == 1

    # 6. OutputTuple: Shared shared dependency (The p_pt3 fix)
    p6 = fx.symbolic_trace(lambda x: (x + 3, (x + 1) * 2))
    assert len(StructMatcher(p6)(p6)) == 1
    # 7. OutputTuple: Independent branches
    p7 = fx.symbolic_trace(lambda x: (x + 1, x + 2))
    assert len(StructMatcher(p7)(p7)) == 1
    # 8. OutputTuple: Cross dependency
    p8 = fx.symbolic_trace(lambda x, y: (x + y, x * y))
    assert len(StructMatcher(p8)(p8)) == 1
    # 9. OutputTuple: Nested tuples
    p9 = fx.symbolic_trace(lambda x: (x + 1, (x + 2, x + 3)))
    t9 = fx.symbolic_trace(lambda x: (x + 1, (x + 2, x + 3), (x + 4)))
    assert len(StructMatcher(p9)(t9)) == 1

    # 10. OutputTuple: Collision/Orphan nodes
    def collision(x, y):
        a, b, c = x + 1, y + 1, x + 2
        return a, c, b

    assert len(StructMatcher(p7)(fx.symbolic_trace(collision))) == 1
    print("All 10 tests passed.")


if __name__ == "__main__":
    test_main()
