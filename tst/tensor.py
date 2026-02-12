from tst.op_registry import op


class Tensor:
    def __init__(self, backend_tensor):
        self._backend_tensor = backend_tensor

    @property
    def backend_tensor(self):
        return self._backend_tensor

    @property
    def shape(self):
        return self.backend_tensor.shape

    @property
    def dtype(self):
        return self.backend_tensor.dtype

    def __repr__(self):
        return f"tst.Tensor({repr(self.backend_tensor)})"

    def __str__(self):
        return f"tst.Tensor({self.backend_tensor})"

    def cast(self, dtype):
        # to be implemented!
        return self

    @op("getitem")
    def __getitem__(self, index):
        ...

    @op("padding")
    def padding(self, target_shape):
        ...

    @op("permute")
    def permute(self, dims):
        ...

    @op("reshape")
    def reshape(self, shape):
        ...

    @op("view")
    def view(self, *shape):
        ...

    @op("assign")
    def assign(self, other):
        ...

    @op("add")
    def __add__(self, other):
        ...

    @op("sum")
    def sum(self, dim=None, keepdim=False):
        ...


def wrap_tensor(tensor):
    if isinstance(tensor, (tuple, list)):
        return [wrap_tensor(t) for t in tensor]
    elif isinstance(tensor, Tensor):
        return tensor
    else:
        # TODO(Xreki): add the check of BackendTensor
        return Tensor(backend_tensor=tensor)


def unwrap_tensor(tensor):
    if isinstance(tensor, (tuple, list)):
        return [unwrap_tensor(t) for t in tensor]
    elif isinstance(tensor, Tensor):
        return tensor.backend_tensor
    else:
        # TODO(Xreki): add the check of BackendTensor
        return tensor


def copy(dst: Tensor, src: Tensor):
    assert src.backend_tensor.numel() == dst.backend_tensor.numel()
    dst.assign(src.reshape(dst.shape))


@op("addmm")
def addmm(input: Tensor, mat1: Tensor, mat2: Tensor, beta: float = 1, alpha: float = 1):
    # out = beta * input + alpha * (mat1 @ mat2)
    ...


@op("matmul")
def matmul(input: Tensor, other: Tensor) -> Tensor:
    # out = input @ other
    ...
