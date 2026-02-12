from collections import defaultdict
import tst
from tst.config import g_execution_mode

# _OP_REGISTRY[op_name][mode] = function
_OP_REGISTRY = defaultdict(dict)


def register_op(op_name, mode):
    def decorator(fn):
        _OP_REGISTRY[op_name][mode] = fn
        return fn

    return decorator


def get_op_impl(op_name, mode):
    if op_name not in _OP_REGISTRY:
        raise KeyError(f"Op not registered: {op_name}")
    if mode not in _OP_REGISTRY[op_name]:
        raise KeyError(f"Op {op_name} not implemented for mode {mode}")
    return _OP_REGISTRY[op_name][mode]


def dispatch_alloc(op_name, *args, **kwargs):
    assert op_name in [
        "alloc_fragment",
        "alloc_shared",
    ], f"Unsupported op name: {op_name}"

    fn = get_op_impl(op_name, g_execution_mode)
    out = fn(*args, **kwargs)
    return tst.Tensor(backend_tensor=out)


def dispatch(op_name, tensor, *args, **kwargs):
    def _wrap(t):
        return t if isinstance(t, tst.Tensor) else tst.Tensor(backend_tensor=t)

    def _unwrap(t):
        return t.backend_tensor if isinstance(t, tst.Tensor) else t

    fn = get_op_impl(op_name, g_execution_mode)
    new_args = tuple(_unwrap(arg) for arg in args)
    new_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
    outs = fn(tensor.backend_tensor, *new_args, **new_kwargs)
    if isinstance(outs, (tuple, list)):
        return tuple(_wrap(out) for out in outs)
    else:
        return _wrap(outs)


def op(op_name):
    def decorator(fn):
        def wrapper(tensor, *args, **kwargs):
            return dispatch(op_name, tensor, *args, **kwargs)

        return wrapper

    return decorator
