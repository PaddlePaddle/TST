import tst


def reduce(func):
    def wrapper(iterators, *args):
        iterator_shape = None
        if isinstance(iterators, tst.Tensor):
            iterator_shape = iterators.shape
        elif isinstance(iterators, (tuple, list)) and isinstance(
            iterators[0], tst.Tensor
        ):
            iterator_shape = iterators[0].shape
        else:
            assert False

        num_iters = None

        assert len(args) > 0
        for i in range(len(args)):
            assert isinstance(args[i], tst.Tensor)
            if len(args[i].shape) > len(iterator_shape):
                num_iters = args[i].shape[0]
        assert num_iters is not None

        for i in range(num_iters):
            args_i = [
                t
                if t.shape == iterator_shape
                else t[(i,) + (slice(None),) * (t.ndim - 1)]
                for t in args
            ]
            func(iterators, *args_i)

    return wrapper
