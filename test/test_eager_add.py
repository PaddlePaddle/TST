import torch
import tst
import unittest
from typing import Callable


def elementwise_add(N, threads, dtype) -> Callable:
    @tst.prim_func
    def main(x: tst.Tensor, y: tst.Tensor) -> tst.Tensor:
        N_roundup = tst.roundup(N, threads)

        with tst.kernel(
            tst.iter_space.block(N_roundup // threads),
            tst.iter_space.thread(threads),
        ) as (bns, _):
            x_local = tst.alloc_fragment([bns], [threads], dtype)
            y_local = tst.alloc_fragment([bns], [threads], dtype)
            out_local = tst.alloc_fragment([bns], [threads], dtype)

            tst.copy(x_local, x.padding([N_roundup]))
            tst.copy(y_local, y.padding([N_roundup]))

            out_local.assign(x_local + y_local)
            return out_local.reshape([N_roundup])[:N]

    return main


class ElementwiseAddEagerTest(unittest.TestCase):
    def test_elementwise_add(self):
        N = 20
        threads = 16
        dtype = "float32"
        add_kernel = elementwise_add(N, threads, dtype)

        x = torch.randn(N, dtype=torch.float32)
        y = torch.randn(N, dtype=torch.float32)
        out = add_kernel(x, y)

        print(f"{x=}")
        print(f"{y=}")
        print(f"{out=}")


if __name__ == "__main__":
    unittest.main()
