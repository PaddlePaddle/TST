from tst.op_registry import dispatch_alloc


def alloc_fragment(iter_indices, shape, dtype):
    return dispatch_alloc("alloc_fragment", iter_indices, shape, dtype)


def alloc_shared(iter_indices, shape, dtype):
    return dispatch_alloc("alloc_shared", iter_indices, shape, dtype)
