import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
import tst


class KernelFuncModule(torch.nn.Module):
    def __init__(self, kernel_func):
        super().__init__()
        self.kernel_func = kernel_func

    def forward(self, *args):
        warpped_args = tst.tensor.wrap_tensor(args)
        return self.kernel_func(*warpped_args)


class GraphBuilder:
    def __init__(self, kernel_func):
        self.kernel_func = kernel_func

    def __call__(self, example_inputs):
        model = KernelFuncModule(self.kernel_func)
        traced_module, _ = self.parse_sole_graph(model, example_inputs)
        ShapeProp(traced_module).propagate(*example_inputs)
        return self._remove_tst_nodes_from_output(traced_module)

    def parse_sole_graph(self, model, inputs):
        traced_module = None
        traced_sample_inputs = None

        def my_backend(gm, sample_inputs):
            nonlocal traced_module
            nonlocal traced_sample_inputs
            traced_module = gm
            traced_sample_inputs = sample_inputs
            return gm.forward

        torch.compile(model, backend=my_backend)(*inputs)
        assert traced_module is not None
        return traced_module, traced_sample_inputs

    def _remove_tst_nodes_from_output(self, gm):
        def is_tst_node(node):
            return node.op.startswith("call_function") and (
                str(node.target).startswith("torch.ops.tst.")
                or str(node.target).startswith("tst.")
            )

        for node in gm.graph.nodes:
            if node.op == "output":
                outs = node.args[0]
                if not isinstance(outs, (tuple, list)):
                    outs = (outs,)

                new_outs = []
                for out in outs:
                    if isinstance(out, fx.Node) and is_tst_node(out):
                        continue
                    new_outs.append(out)

                if len(new_outs) == 1:
                    node.args = (new_outs[0],)
                else:
                    node.args = (tuple(new_outs),)

        gm.graph.lint()
        gm.recompile()
        return gm
