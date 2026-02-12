import torch
from tst.op_registry import register_op
from tst.config import ExecutionMode


def convert_to_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    if not isinstance(dtype, str):
        raise ValueError(f"Unsupported dtype: {dtype}")

    dtype_str2torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float": torch.float32,
        "float32": torch.float32,
        "float64": torch.float64,
        "bool": torch.bool,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }

    key = dtype.lower().replace("torch.", "")
    if key not in dtype_str2torch_dtype:
        raise ValueError(f"Unsupported dtype string: {key}")
    return dtype_str2torch_dtype[key]


@register_op("alloc_fragment", ExecutionMode.EAGER)
def alloc_fragment(iter_indices, shape, dtype):
    iter_indice_tensors = [ii.node for ii in iter_indices]
    tensor = torch.ops.tst.alloc_fragment(
        iter_indice_tensors, shape, convert_to_torch_dtype(dtype)
    )
    return tensor


@register_op("alloc_shared", ExecutionMode.EAGER)
def alloc_shared(iter_indices, shape, dtype):
    iter_indice_tensors = [ii.node for ii in iter_indices]
    tensor = torch.ops.tst.alloc_shared(
        iter_indice_tensors, shape, convert_to_torch_dtype(dtype)
    )
    return tensor
