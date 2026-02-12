from typing import Callable

import tst
from tst.eager.graph_builder import GraphBuilder


class PrimFunc:
    def __init__(self, kernel_func: Callable):
        self._kernel_func = kernel_func
        self._args = None
        self._graph_module = None

    def __call__(self, *args):
        self._args = tst.tensor.wrap_tensor(args)
        outs = self._kernel_func(*self._args)
        return outs

    @property
    def graph_module(self):
        if not self._graph_module:
            assert (
                self._args is not None
            ), "Need to run the prim_func once to trigger graph capture."
            self.make_graph(*self._args)
        return self._graph_module

    def make_graph(self, *args):
        backend_tensor_args = tst.tensor.unwrap_tensor(args)
        self._graph_module = GraphBuilder(self._kernel_func)(backend_tensor_args)


def prim_func(fn):
    return PrimFunc(fn)
