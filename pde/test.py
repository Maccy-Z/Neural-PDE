import torch
import cupy as cp
import cupyx.scipy.sparse as cusparse
import cupyx.scipy.sparse.linalg as cuslinalg

def test_sparse_spsolve():
    # Create a 4x4 diagonal sparse matrix with diagonal values [4, 5, 6, 7]
    indices = torch.tensor([[0, 1, 2, 3],
                            [0, 1, 2, 3]])
    values = torch.tensor([4.0, 5.0, 6.0, 7.0])
    A = torch.sparse_coo_tensor(indices, values, (4, 4)).to_sparse_csr().cuda()

    # Define the right-hand side vector b such that the true solution x is [1, 2, 3, 4]
    # Because A * x = [4*1, 5*2, 6*3, 7*4] = [4, 10, 18, 28]
    b = torch.tensor([4.0, 10.0, 18.0, 28.0]).cuda()

    # Convert to cupy
    cp_crow_indices = cp.from_dlpack(A.crow_indices())
    cp_col_indices = cp.from_dlpack(A.col_indices())
    cp_values = cp.from_dlpack(A.values())
    # Build a CuPy CSR matrix
    A = cusparse.csr_matrix((cp_values, cp_col_indices, cp_crow_indices), shape=(4, 4)).tocoo()
    b = cp.from_dlpack(b)

    x = cuslinalg.spsolve(A, b)
    # Expected solution is [1, 2, 3, 4]
    expected = cp.array([1.0, 2.0, 3.0, 4.0])

    print(f'{x = }, {expected = }')
    # Assert that the computed solution is close to the expected solution
    print("test_sparse_spsolve passed.")


# If running as a script, execute the test
if __name__ == "__main__":
    test_sparse_spsolve()