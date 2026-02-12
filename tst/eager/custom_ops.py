import torch
from torch.library import Library


lib = Library("tst", "DEF")
lib.define("kernel(Tensor anchor) -> Tensor")
lib.define("iter_space(Tensor kernel, int n, str kind) -> Tensor")
lib.define("iter_space_domain(Tensor[] iter_spaces) -> Tensor")
lib.define(
    "alloc_fragment(Tensor[] iter_spaces, int[] shape, ScalarType dtype) -> Tensor"
)
lib.define(
    "alloc_shared(Tensor[] iter_spaces, int[] shape, ScalarType dtype) -> Tensor"
)


def kernel_impl_meta(anchor: torch.Tensor):
    return torch.empty((), device="meta")


def kernel_impl_cpu(anchor: torch.Tensor):
    return torch.empty((), device="cpu")


def kernel_impl_cuda(anchor: torch.Tensor):
    return torch.empty((), device="cuda")


def iter_space_impl(kernel: torch.Tensor, n: int, kind: str):
    return torch.empty(n, device=kernel.device)


def _span_iter_space_shape(iter_spaces: list[torch.Tensor]):
    return [s for t in iter_spaces for s in t.size()]


def iter_space_domain_impl(iter_spaces: list[torch.Tensor]):
    assert len(iter_spaces) > 0
    spanned_shape = _span_iter_space_shape(iter_spaces)
    return torch.empty(spanned_shape, device=iter_spaces[0].device)


def alloc_impl_meta(
    iter_spaces: list[torch.Tensor], shape: list[int], dtype: torch.dtype
):
    full_shape = _span_iter_space_shape(iter_spaces) + shape
    return torch.empty(full_shape, dtype=dtype, device="meta")


def alloc_impl_cpu(
    iter_spaces: list[torch.Tensor], shape: list[int], dtype: torch.dtype
):
    full_shape = _span_iter_space_shape(iter_spaces) + shape
    return torch.empty(full_shape, dtype=dtype, device="cpu")


def alloc_impl_cuda(
    iter_spaces: list[torch.Tensor], shape: list[int], dtype: torch.dtype
):
    full_shape = _span_iter_space_shape(iter_spaces) + shape
    return torch.empty(full_shape, dtype=dtype, device="cuda")


CUSTOM_OPS = {
    "Meta": {
        "kernel": kernel_impl_meta,
        "iter_space": iter_space_impl,
        "iter_space_domain": iter_space_domain_impl,
        "alloc_fragment": alloc_impl_meta,
        "alloc_shared": alloc_impl_meta,
    },
    "CPU": {
        "kernel": kernel_impl_cpu,
        "iter_space": iter_space_impl,
        "iter_space_domain": iter_space_domain_impl,
        "alloc_fragment": alloc_impl_cpu,
        "alloc_shared": alloc_impl_cpu,
    },
    "CUDA": {
        "kernel": kernel_impl_cuda,
        "iter_space": iter_space_impl,
        "iter_space_domain": iter_space_domain_impl,
        "alloc_fragment": alloc_impl_cuda,
        "alloc_shared": alloc_impl_cuda,
    },
}


def register_all_custom_ops(lib_impl):
    for device_type, ops in CUSTOM_OPS.items():
        for name, func in ops.items():
            lib_impl.impl(name, func, device_type)


lib_impl = Library("tst", "IMPL")
register_all_custom_ops(lib_impl)
