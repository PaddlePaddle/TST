import torch
import tst
import unittest
import numpy as np


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads=32,
    dtype="float16",
    accum_dtype="float32",
):
    @tst.prim_func
    def main(A: tst.Tensor, B: tst.Tensor) -> tst.Tensor:
        M_roundup = tst.roundup(M, block_M)
        N_roundup = tst.roundup(N, block_N)
        K_roundup = tst.roundup(K, block_K)

        with tst.kernel(
            tst.iter_space.block(M_roundup // block_M),
            tst.iter_space.block(N_roundup // block_N),
            tst.iter_space.thread(threads),
        ) as (bms, bns, _):
            C_local = tst.alloc_fragment([bms, bns], [block_M, block_N], accum_dtype)

            with tst.iter_space.pipelined(K_roundup // block_K, num_stages=3) as ks:
                A_shared = tst.alloc_shared([bms, ks], [block_M, block_K], dtype)
                B_shared = tst.alloc_shared([bns, ks], [block_K, block_N], dtype)

                # A: [bms, block_M, ks, block_K] -> A_shared: [bms, ks, block_M, block_K]
                tst.copy(
                    A_shared.permute([0, 2, 1, 3]), A.padding([M_roundup, K_roundup])
                )
                # B: [ks, block_K, bns, block_N] -> B_shared: [bns, ks, block_K, block_N]
                tst.copy(
                    B_shared.permute([1, 2, 0, 3]), B.padding([K_roundup, N_roundup])
                )

                # [bms, 1, ks, block_M, block_K] * [1, bns, ks, block_N, block_N]
                # -> [bms, bns, ks, block_M, block_N]
                # -> C_local: [bms, bns, block_M, block_N]
                C_local.assign(
                    tst.matmul(
                        A_shared[:, None, :, :, :], B_shared[None, :, :, :, :]
                    ).sum(dim=2)
                )

            # [bms, bns, block_M, block_N] -> [bms, block_M, bns, block_N]
            return C_local.permute([0, 2, 1, 3]).reshape((M_roundup, N_roundup))[:M, :N]

    return main


class MatmulEagerTest(unittest.TestCase):
    def test_matmul(self):
        M = 32
        N = 32
        K = 32
        block_M = 16
        block_N = 16
        block_K = 16
        dtype = "float32"
        matmul_kernel = matmul(M, N, K, block_M, block_N, block_K, dtype=dtype)

        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        C = matmul_kernel(A, B)
        print(matmul_kernel.graph_module.graph)

        C_ref = torch.matmul(A, B)
        np.testing.assert_allclose(
            C.backend_tensor.numpy(), C_ref.numpy(), atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
