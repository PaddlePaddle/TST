import torch


def kernel():
    anchor = torch.empty(())
    return torch.ops.tst.kernel(anchor)


def block(kernel, n):
    return torch.ops.tst.iter_space(kernel, n, "block")


def thread(kernel, n):
    return torch.ops.tst.iter_space(kernel, n, "thread")


def pipelined(kernel, n):
    return torch.ops.tst.iter_space(kernel, n, "pipelined")


def iter_space_domain(iter_indices):
    iter_indice_tensors = [ii.node for ii in iter_indices]
    return torch.ops.tst.iter_space_domain(iter_indice_tensors)
