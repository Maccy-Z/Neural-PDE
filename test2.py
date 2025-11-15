import cupy as cp
import cupyx.scipy.sparse as sp
import time
import threading

import nvmath

cp.random.seed(0)
# The number of equations.
n = 5500

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = sp.random(n, n, density=0.01, format="csr", dtype="float32")
a += sp.diags([2.0] * n, format="csr", dtype="float32")
print(f'{a.nnz = }')
new_data = cp.random.rand(a.nnz, dtype="float32")  # New values for the non-zero entries
a2 = B = sp.csr_matrix(
    (new_data, a.indices.copy(), a.indptr.copy()),
    shape=a.shape,
)
a2 += sp.diags([2.0] * n, format="csr", dtype="float32")

# Create the RHS, which can be a matrix or vector in column-major layout.
b = cp.ones((n,), dtype="float32")

print(f'{a.data.std() = }')
print(f'{a2.data.std() = }')
print()
# Solve a @ x = b for x.
config = nvmath.sparse.advanced.DirectSolverOptions(multithreading_lib="/home/maccyz/miniforge3/envs/test/lib/python3.13/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0")


# Use the stateful object as a context manager to automatically release resources.

# # 1) Full sparse solve
cp.cuda.get_current_stream().synchronize()
st = time.time()
solver = nvmath.sparse.advanced.DirectSolver(a, b, options=config)
solver.plan()
solver.factorize()
x = solver.solve()
cp.cuda.get_current_stream().synchronize()
print(f'Full solve time: {time.time() - st:.4f} seconds')

# 2) Now let's modify the LHS. For small changes, the LHS can be modified and iterative refinement
# # Update A in place.
# a.data *= 1.1
solution_config = solver.solution_config
solution_config.ir_num_steps = 100
# cp.cuda.get_current_stream().synchronize()
# st = time.time()
# x = solver.solve()
# cp.cuda.get_current_stream().synchronize()
# print(f"Just solve time: {time.time() - st:.4f} seconds")
# y = (a @ x) - b
# print(f"Residual norm after refactorization: {cp.linalg.norm(y):.4e}")
# print(x)
# print(solver.solution_config.solution_algorithm)
# print()


# 3) For larger changes to A, it's better to refactorize.
solution_config.ir_num_steps = 0

a.data[...] = a2.data

cp.cuda.get_current_stream().synchronize()
st = time.time()
solver.factorize()
x = solver.solve()
cp.cuda.get_current_stream().synchronize()
print(f"Refactorize time: {time.time() - st:.4f} seconds")

y = (a @ x) - b
print(f"Residual norm after refactorization: {cp.linalg.norm(y):.4e}")
print(x)
