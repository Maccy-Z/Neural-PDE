import ctypes
from cupy_backends.cuda.libs import cusolver
import cupy as cp
import numpy as np
# print(vars(cusolver).keys())
# lib = cusolver._lib            # <class 'ctypes.CDLL'>
# #
# # print(lib)


lib = ctypes.CDLL('/home/maccyz/miniforge3/envs/neural_pde/lib/python3.11/site-packages/cupy_backends/cuda/libs/cusolver.cpython-311-x86_64-linux-gnu.so')

bufSize_fn = getattr(lib, 'cusolverDnSXgels_bufferSize')
# Common aliases
c_int       = ctypes.c_int
c_size_t    = ctypes.c_size_t
c_void_p    = ctypes.c_void_p

bufSize_fn.restype  = c_int
bufSize_fn.argtypes = [
    c_void_p,   # cusolverDnHandle_t handle
    c_int,      # jobz  ('V'==1, 'N'==0)
    c_int,      # uplo  ('U'==0, 'L'==1)  – follows CUSOLVER ENUM docs
    c_int,      # n
    c_void_p,   # double* A (device ptr)
    c_int,      # lda
    c_void_p    # int*  lwork (host ptr)
]

handle = cusolver.create()                  # returns cusolverDnHandle_t
stream = cp.cuda.get_current_stream()       # default stream CuPy is on
cusolver.setStream(handle, stream.ptr)      # keep everything on same stream

# Example matrix (Hermitian, 4×4, column-major on device)
A  = cp.asarray(np.array([[1.,2.,3.,4.],
                          [2.,5.,6.,7.],
                          [3.,6.,8.,9.],
                          [4.,7.,9.,10.]], order='F'))

lwork_host = c_size_t()   # output parameter – host memory

status = bufSize_fn(
    handle,
    4,                     # jobz = 1 ⇒ compute eigenvectors
    4,                     # uplo = 0 ⇒ use upper triangle
    1,

    c_void_p(A.data.ptr),  # device pointer to matrix
    n,                     # leading dimension
    ctypes.byref(lwork_host)
)

if status != 0:
    raise RuntimeError(f"CUSOLVER error {status}")
print("Workspace size (bytes):", lwork_host.value)




