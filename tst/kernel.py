from tst import iter_space
from tst.eager import iter_space_impl


class _TraceState:
    current_kernel = None


g_trace_state = _TraceState()


class KernelContext:
    def __init__(self, *iter_spaces):
        for s in iter_spaces:
            if not isinstance(s, (iter_space.Block, iter_space.Thread)):
                raise TypeError(f"Unknown iter_space type: {type(s)}")

        self.iter_spaces = iter_spaces
        self.grid_dim = [
            s.n for s in self.iter_spaces if isinstance(s, iter_space.Block)
        ]
        self.block_dim = [
            s.n for s in self.iter_spaces if isinstance(s, iter_space.Thread)
        ]
        assert len(self.grid_dim) > 0 and len(self.grid_dim) <= 3, f"{self.grid_dim=}"
        assert (
            len(self.block_dim) > 0 and len(self.block_dim) <= 3
        ), f"{self.block_dim=}"

    def __enter__(self):
        self.kernel_node = iter_space_impl.kernel()
        for space in self.iter_spaces:
            space.make_graph_node(self.kernel_node)
        g_trace_state.current_kernel = self.kernel_node
        return tuple(self.iter_spaces)

    def __exit__(self, exc_type, exc_val, exc_tb):
        g_trace_state.current_kernel = None
        return False


def kernel(*iter_spaces):
    return KernelContext(*iter_spaces)
