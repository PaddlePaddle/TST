import torch
import torch.fx as fx
from typing import Dict, List, Set, Union, Tuple, Optional, Callable
from dataclasses import dataclass

# ImportFrom sources (Logical Reference)
from tst.torch_ap.struct_matcher import StructMatcher
from torch.fx.passes.infra.pass_manager import PassResult

MatchResult = Dict[fx.Node, fx.Node]


class MatchContext:
    def __init__(
        self, match_result: MatchResult, target: fx.GraphModule, pattern: fx.GraphModule
    ):
        self.match_result = match_result
        self.target = target
        self.pattern = pattern


def _dead_code_eliminate(gm: fx.GraphModule) -> fx.GraphModule:
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


class StructPatternReplacer:
    def __init__(
        self,
        pattern: fx.GraphModule,
        get_replacement: Callable[[MatchContext], fx.GraphModule],
        constraint: Callable[[MatchContext], bool] = lambda _: True,
    ):
        self.pattern = pattern
        self.get_replacement = get_replacement
        self.constraint = constraint
        self.matcher = StructMatcher(pattern)

    def __call__(self, target: fx.GraphModule) -> PassResult:
        modified = False
        matches: List[MatchResult] = self.matcher(target)
        consumed_nodes: Set[fx.Node] = set()
        processed_sigs: Set[Tuple[int, ...]] = set()

        for m_res in matches:
            target_nodes = {v for k, v in m_res.items() if k.op != "placeholder"}
            m_sig = tuple(sorted(id(n) for n in target_nodes))
            if m_sig in processed_sigs or not target_nodes.isdisjoint(consumed_nodes):
                continue

            ctx = MatchContext(m_res, target, self.pattern)
            if not self.constraint(ctx):
                continue

            self._replace_subgraph(target, m_res, ctx)
            processed_sigs.add(m_sig)
            consumed_nodes.update(target_nodes)
            modified = True

        if modified:
            _dead_code_eliminate(target)
        return PassResult(target, modified)

    def _replace_subgraph(
        self, target_gm: fx.GraphModule, match: MatchResult, ctx: MatchContext
    ):
        replacement_gm: fx.GraphModule = self.get_replacement(ctx)
        symbol_env: Dict[fx.Node, fx.Node] = {}

        p_phs = [n for n in self.pattern.graph.nodes if n.op == "placeholder"]
        r_phs = [n for n in replacement_gm.graph.nodes if n.op == "placeholder"]
        for p_ph, r_ph in zip(p_phs, r_phs):
            symbol_env[r_ph] = match[p_ph]

        p_out = next(n for n in self.pattern.graph.nodes if n.op == "output")
        p_out_args = p_out.args[0]
        first_p_sink = (
            p_out_args[0] if isinstance(p_out_args, (list, tuple)) else p_out_args
        )
        anchor_node = match[first_p_sink]

        with target_gm.graph.inserting_before(anchor_node):
            for r_node in replacement_gm.graph.nodes:
                if r_node.op == "placeholder":
                    continue
                if r_node.op == "output":
                    r_outs = (
                        r_node.args[0]
                        if isinstance(r_node.args[0], (list, tuple))
                        else [r_node.args[0]]
                    )
                    p_outs = (
                        p_out_args
                        if isinstance(p_out_args, (list, tuple))
                        else [p_out_args]
                    )
                    for ps, rs in zip(p_outs, r_outs):
                        match[ps].replace_all_uses_with(symbol_env[rs])
                    continue
                new_args = fx.node.map_arg(
                    r_node.args,
                    lambda n: symbol_env[n] if isinstance(n, fx.Node) else n,
                )
                new_kwargs = fx.node.map_arg(
                    r_node.kwargs,
                    lambda n: symbol_env[n] if isinstance(n, fx.Node) else n,
                )
                symbol_env[r_node] = target_gm.graph.create_node(
                    r_node.op, r_node.target, args=new_args, kwargs=new_kwargs
                )


def test_main():
    # Setup shared patterns
    p_add = fx.symbolic_trace(lambda x: torch.add(x, 1.0))
    repl_mul = lambda ctx: fx.symbolic_trace(lambda x: torch.mul(x, 10.0))
    replacer = StructPatternReplacer(p_add, repl_mul)

    # 1. Multi-match (Standard)
    t1 = fx.symbolic_trace(lambda x, y: torch.add(x, 1.0) - torch.add(y, 1.0))
    assert (
        replacer(t1).modified
        and len([n for n in t1.graph.nodes if n.target == torch.mul]) == 2
    )
    print("Test 1: Multi-match Passed")

    # 2. Overlap Protection (Standard)
    t2 = fx.symbolic_trace(lambda x: torch.add(torch.add(x, 1.0), 1.0))
    assert (
        replacer(t2).modified
        and len([n for n in t2.graph.nodes if n.target == torch.mul]) == 1
    )
    print("Test 2: Overlap Protection Passed")

    # 3. Name Constraint
    def name_c(ctx):
        return (
            "target"
            in ctx.match_result[
                next(n for n in ctx.pattern.graph.nodes if n.op == "placeholder")
            ].name
        )

    t3 = fx.symbolic_trace(
        lambda target, other: torch.add(target, 1.0) + torch.add(other, 1.0)
    )
    assert (
        StructPatternReplacer(p_add, repl_mul, name_c)(t3).modified
        and len([n for n in t3.graph.nodes if n.target == torch.mul]) == 1
    )
    print("Test 3: Name Constraint Passed")

    # 4. Multi-Input Pattern
    p4 = fx.symbolic_trace(lambda x, y: x + y)
    t4 = fx.symbolic_trace(lambda a, b, c: (a + b) + c)
    assert StructPatternReplacer(p4, lambda ctx: fx.symbolic_trace(lambda x, y: x * y))(
        t4
    ).modified
    print("Test 4: Multi-Input Pattern Passed")

    # 5. Dead Code Elimination (Implicit)
    t5 = fx.symbolic_trace(lambda x: torch.add(x, 1.0))
    replacer(t5)
    assert not any(n.target == torch.add for n in t5.graph.nodes)
    print("Test 5: DCE Verification Passed")

    # 6. Negative Match (Structure)
    t6 = fx.symbolic_trace(lambda x: torch.sub(x, 1.0))
    assert not replacer(t6).modified
    print("Test 6: Negative Match Passed")

    # 7. Attribute Access Pattern
    class MockMod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = 0.5

        def forward(self, x):
            return x + self.alpha

    p7 = fx.symbolic_trace(MockMod())
    t7 = fx.symbolic_trace(MockMod())
    assert StructPatternReplacer(p7, repl_mul)(t7).modified
    print("Test 7: Attribute Access Passed")

    # 8. Non-Tensor Argument Preservation
    p8 = fx.symbolic_trace(lambda x: torch.mean(x, dim=1))
    t8 = fx.symbolic_trace(lambda x: torch.mean(x, dim=1))
    assert StructPatternReplacer(
        p8, lambda ctx: fx.symbolic_trace(lambda x: torch.sum(x, dim=1))
    )(t8).modified
    print("Test 8: Keyword Arg Preservation Passed")

    # 9. Identity Replacement (No-op check)
    t9 = fx.symbolic_trace(lambda x: torch.add(x, 1.0))
    assert StructPatternReplacer(p_add, lambda ctx: p_add)(
        t9
    ).modified  # Modified because nodes are replaced, though functionally same
    print("Test 9: Identity Replacement Passed")

    # 10. Multi-Output pattern (Tuple)
    p10 = fx.symbolic_trace(lambda x: (x + 1.0, x + 2.0))
    t10 = fx.symbolic_trace(lambda x: (x + 1.0, x + 2.0, x + 3.0))
    repl10 = lambda ctx: fx.symbolic_trace(lambda x: (x * 10.0, x * 20.0))
    assert StructPatternReplacer(p10, repl10)(t10).modified
    print("Test 10: Multi-Output Pattern Passed")


if __name__ == "__main__":
    test_main()
