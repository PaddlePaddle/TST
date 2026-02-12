from enum import Enum
from tst.kernel import g_trace_state
from tst.eager import iter_space_impl


class IterSpaceKind(str, Enum):
    BLOCK = "block"
    THREAD = "thread"
    PIPELINED = "pipelined"


class IterSpaceBase:
    def __init__(self, kind, n):
        self.kind = kind
        self.n = n
        self.iters = list(range(n))
        self.node = None

    def make_graph_node(self, kernel):
        raise NotImplementedError("make_graph_node is not implemented yet!")

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"iter_space.{self.kind}<{self.n}>"


class Block(IterSpaceBase):
    def __init__(self, n):
        super().__init__(IterSpaceKind.BLOCK, n)

    def make_graph_node(self, kernel):
        self.node = iter_space_impl.block(kernel, self.n)


class Thread(IterSpaceBase):
    def __init__(self, n):
        super().__init__(IterSpaceKind.THREAD, n)

    def make_graph_node(self, kernel):
        self.node = iter_space_impl.thread(kernel, self.n)


class Pipelined(IterSpaceBase):
    def __init__(self, n):
        super().__init__(IterSpaceKind.PIPELINED, n)

    def make_graph_node(self, kernel):
        self.node = iter_space_impl.pipelined(kernel, self.n)


def block(n: int):
    return Block(n)


def thread(n: int):
    return Thread(n)


class PipelinedContext:
    def __init__(self, n, num_stages):
        self.iter_space = Pipelined(n)
        self.num_stages = num_stages

    def __enter__(self):
        kernel_node = g_trace_state.current_kernel
        self.iter_space.make_graph_node(kernel_node)
        return self.iter_space

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def pipelined(n: int, num_stages: int):
    return PipelinedContext(n, num_stages)
