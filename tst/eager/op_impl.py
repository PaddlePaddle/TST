import torch
from tst.op_registry import register_op
from tst.config import ExecutionMode


@register_op("view", ExecutionMode.EAGER)
def view(input: torch.Tensor, *shape):
    return input.view(*shape)


@register_op("reshape", ExecutionMode.EAGER)
def reshape(input: torch.Tensor, shape: int | list):
    return input.reshape(shape)


@register_op("permute", ExecutionMode.EAGER)
def permute(input: torch.Tensor, dims: int | list):
    return input.permute(dims)


@register_op("getitem", ExecutionMode.EAGER)
def getitem(input: torch.Tensor, index) -> torch.Tensor:
    return input[index]


@register_op("padding", ExecutionMode.EAGER)
def padding(input: torch.Tensor, target_shape: list) -> torch.Tensor:
    assert input.dim() == len(
        target_shape
    ), f"the length of target_shape ({len(target_shape)}) != {input.dim()}"
    pad = []
    for i in range(len(target_shape) - 1, -1, -1):
        current_dim = input.size(i)
        target_dim = target_shape[i]
        if target_dim < current_dim:
            raise ValueError(
                f"target_shape {target_shape} must >= input shape {tuple(input.shape)}"
            )
        pad.extend([0, target_dim - current_dim])
    if all(d == 0 for d in pad):
        return input
    else:
        return torch.nn.functional.pad(input, pad, mode="constant", value=0)


@register_op("assign", ExecutionMode.EAGER)
def assign(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return dst.copy_(src)


@register_op("add", ExecutionMode.EAGER)
def add(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return input + other


@register_op("sum", ExecutionMode.EAGER)
def sum(
    input: torch.Tensor,
    dim: int | list = None,
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.sum(input, dim, keepdim=keepdim)


@register_op("addmm", ExecutionMode.EAGER)
def addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    beta: float = 1,
    alpha: float = 1,
) -> torch.Tensor:
    # out = beta * input + alpha * (mat1 @ mat2)
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


@register_op("matmul", ExecutionMode.EAGER)
def matmul(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    # out = input @ other
    return torch.matmul(input, other)
