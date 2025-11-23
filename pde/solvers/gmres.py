import numpy as np
import cupy
import cupy as cp
#
from cupy import cublas
from cupyx.scipy.sparse.linalg._iterative import _make_system  # , #_make_compute_hu
from cupy_backends.cuda.libs import cublas as _cublas
from cupy.cuda import device
from cupyx.scipy.linalg import solve_triangular
import torch

from codetiming import Timer

def _make_compute_hu(V):
    handle = device.get_cublas_handle()
    if V.dtype.char == 'f':
        gemv = _cublas.sgemv
    elif V.dtype.char == 'd':
        gemv = _cublas.dgemv
    elif V.dtype.char == 'F':
        gemv = _cublas.cgemv
    elif V.dtype.char == 'D':
        gemv = _cublas.zgemv
    n = V.shape[0]
    one = np.array(1.0, V.dtype)
    zero = np.array(0.0, V.dtype)
    mone = np.array(-1.0, V.dtype)

    # def compute_hu(u, j, H):
    #     # Gramm-Schmidt process
    #     # Instead of allocating a new h array, use the column of H directly
    #     h_col = H[:j+1, j]  # View of column j in H matrix
    #
    #     # Compute V[:, :j+1].conj().T @ u and store in h_col
    #     gemv(handle, _cublas.CUBLAS_OP_C, n, j+1, one.ctypes.data, V.data.ptr,
    #          n, u.data.ptr, 1, zero.ctypes.data, h_col.data.ptr, 1)
    #
    #     # Compute u = u - V[:, :j+1] @ h_col
    #     gemv(handle, _cublas.CUBLAS_OP_N, n, j+1, mone.ctypes.data, V.data.ptr,
    #          n, h_col.data.ptr, 1, one.ctypes.data, u.data.ptr, 1)
    #     return u

    def compute_hu(u, j, H):
        # working buffer for the projection coefficients of *this* pass
        tmp = cp.empty(j + 1, dtype=u.dtype)

        h_col = H[:j + 1, j]  # view on the target column – accumulates the sum

        # print(j)
        for it in range(2):
            # tmp  = V[:, :j+1]^H  u
            _cublas.sgemv(handle, _cublas.CUBLAS_OP_T,
                        n, j + 1,
                        one.ctypes.data, V.data.ptr, n,
                        u.data.ptr, 1,
                        zero.ctypes.data, tmp.data.ptr, 1)

            # accumulate into H  (beta = 1 after the first pass)
            if it == 0:
                h_col[...] = tmp
            else:
                h_col += tmp  # H ← H + tmp

            # u  = u − V[:, :j+1]  tmp
            _cublas.sgemv(handle, _cublas.CUBLAS_OP_N,
                        n, j + 1,
                        mone.ctypes.data, V.data.ptr, n,
                        tmp.data.ptr, 1,
                        one.ctypes.data, u.data.ptr, 1)

            # print(cp.linalg.norm(tmp))
            # if cp.linalg.norm(tmp):
                # break
            if j > 500:
                break

        return u

    return compute_hu

def lstsq_sxgels(A, B):
    """
    Least–squares solve  (min ‖AX – B‖₂)  using cuSOLVER `cusolverDnSXgels`.

    Parameters
    ----------
    A : (m, n) cupy.ndarray, float32
        Coefficient matrix.  Only **float32** is accepted; the kernel
        runs in TF32 / mixed precision internally.
    B : (m,) or (m, nrhs) cupy.ndarray, float32
        Right-hand side vector(s).
    Returns
    -------
    X : (n,) or (n, nrhs) cupy.ndarray, float32
        Least-squares solution.

    """

    from cupy_backends.cuda.libs import cusolver
    import ctypes

    m, n = A.shape

    # RHS normalisation -----------------------------------------------------
    B = B[:, None] if B.ndim == 1 else B
    nrhs = 1

    # Fortran contiguous copies (cusolver works column-major) ---------------
    X_f = cp.empty((n, nrhs), dtype=cp.float32, order="F")

    lda, ldb, ldx = m, m, n

    # cuSOLVER handle --------------------------------------------------------
    handle = device.get_cusolver_handle() #_cusolver.cusolverDnCreate()

    # Workspace query --------------------------------------------------------
    result = cusolver.ssgels_bufferSize(           # :contentReference[oaicite:0]{index=0}
        handle, m, n, nrhs,
        int(A.data.ptr), lda,
        int(B.data.ptr), ldb,
        int(X_f.data.ptr), ldx,
        0,
    )

    work = cp.empty(result)

    # Device info -----------------------------------------------------------
    dev_info = cp.empty(1, dtype=cp.int32)

    # Solve -----------------------------------------------------------------
    out = cusolver.ssgels(                      # :contentReference[oaicite:1]{index=1}
        handle, m, n, nrhs,
        int(A.data.ptr), lda,
        int(B.data.ptr), ldb,
        int(X_f.data.ptr), ldx,
        int(work.data.ptr), result,
        dev_info.data.ptr,
    )
    info = int(dev_info.get())
    if info < 0:
        raise RuntimeError(
            f"cusolverDnSXgels: argument {-info} was invalid (info={info})")

    return X_f.ravel()

def lstsq_qr(A, b):
    # A : (m, n) with m >= n
    # b : (m,) or (m, k)
    # returns x of shape (n,) or (n, k)
    Q, R = cupy.linalg.qr(A, mode='reduced')           # Q: (m,n), R: (n,n)
    y    = Q.T.conj() @ b                             # project b onto col(Q)
    x    = solve_triangular(R, y, lower=False)       # solve R x = y

    return x

def gmres(A, b, x0=None, rtol=1e-5, restart=None, maxiter=None, M=None, atol=None):
    """Uses Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        restart (int): Number of iterations between restarts. Larger values
            increase iteration cost, but may be necessary for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call on every restart.
            It is called as ``callback(arg)``, where ``arg`` is selected by
            ``callback_type``.
        callback_type (str): 'x' or 'pr_norm'. If 'x', the current solution
            vector is used as an argument of callback function. if 'pr_norm',
            relative (preconditioned) residual norm is used as an argument.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    Reference:
        M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
        Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).

    .. seealso:: :func:`scipy.sparse.linalg.gmres`
    """

    A, M, x, b = _make_system(A, M, x0, b)
    A_matvec = A.matvec
    #psolve = M.matvec

    n = A.shape[0]
    b_norm = cupy.linalg.norm(b)

    if atol is None:
        atol = rtol * float(b_norm)
    else:
        atol = max(float(atol), rtol * float(b_norm))

    restart = min(restart, n)

    V = cupy.empty((n, restart), dtype=A.dtype, order='F')
    H = cupy.zeros((restart+1, restart), dtype=A.dtype, order='F')
    e_gpu = cupy.zeros((restart+1,), dtype=A.dtype)

    compute_hu = _make_compute_hu(V)

    for full_iters in range(maxiter // restart):
        r = b - A_matvec(x)
        r_norm = cublas.nrm2(r)

        if r_norm <= atol:
            break
        v = r / r_norm
        V[:, 0] = v
        e_gpu[0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            u = A_matvec(v)
            # Pass H to compute_hu to use its columns directly
            u = compute_hu(u, j, H)

            cublas.nrm2(u, out=H[j+1, j])
            if j+1 < restart:
                v = u / H[j+1, j]
                V[:, j+1] = v

        # Solve the least squares problem using CuPy
        y = lstsq_qr(H, e_gpu)
        # y = lstsq_sxgels(H, e_gpu)
        x += V @ y


    r_norm = cupy.linalg.norm(b - A_matvec(x))
    # print()
    # print(f'{r_norm = }, {final_residual_norm = }')
    info = {'completed': (r_norm <= atol), 'iters': full_iters, 'resid_norm': r_norm, 'frac_acc': r_norm/b_norm}

    return x, info


# Custom GMRES implementation
def gmres_cust(A, b, x0=None, rtol=1e-5, maxiter=1000, restart=None):
    """
    GMRES algorithm to solve the linear system Ax = b.

    Arguments:
    A : callable or tensor
        If callable, it should be a function that returns the product Ax for a given x.
        If tensor, it should be a 2D sparse tensor representing the matrix A.
    b : tensor
        The right-hand side vector.
    x0 : tensor, optional
        Initial guess for the solution (default is None, which means x0 = 0).
    tol : float, optional
        Tolerance for convergence (default is 1e-5).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).
    restart : int, optional
        Restart parameter (default is None, which means no restart).

    Returns:
    x : tensor
        Approximate solution to the system Ax = b.
    info : dict
        Dictionary containing information about the convergence (e.g., number of iterations).
    """
    device = b.device
    dtype = b.dtype
    n = b.shape[0]

    A_in = A
    if x0 is None:
        x0 = torch.zeros_like(b)

    if restart is None:
        restart = n  # No restart if not specified

    def Ain_vec(v):
        return A_in @ v

    # Initialize
    x = x0

    b_norm = torch.norm(b)
    r = b - Ain_vec(x)
    beta = torch.norm(r)
    eye = torch.eye(restart + 1, dtype=dtype, device=device)
    # if beta < tol:
    #     return x, {'converged': True, 'iterations': 0, 'residual_norm': beta.item()}

    Q = torch.zeros((n, restart + 1), dtype=dtype, device=device)
    H = torch.zeros((restart + 1, restart), dtype=dtype, device=device)

    Q[:, 0] = r / beta

    residual_norm = None
    for k in range(maxiter // restart):
        for j in range(restart):
            v = Ain_vec(Q[:, j])
            # Vectorized computation of H[0:j+1, j]
            H[:j+1, j] = torch.mv(Q[:, :j+1].T, v)
            # Vectorized update of v
            v -= torch.mv(Q[:, :j+1], H[:j+1, j])

            H[j+1, j] = torch.norm(v)
            Q[:, j+1] = v / H[j+1, j]

        # Solve the least squares problem H * y = beta * e1
        eye[0] = beta
        y = torch.linalg.lstsq(H, eye)
        y = y.solution

        # Update the solution
        x += torch.mv(Q[:, :restart], y[:, 0])

        # Calculate the new residual
        r = b - Ain_vec(x)
        residual_norm = torch.norm(r)

        # Exit if converged
        err_ratio = residual_norm / b_norm
        if err_ratio < rtol:
            return x, {'converged': True, 'iterations': k * restart + j + 1, 'residual_norm': residual_norm}

        # Restart with new initial residual
        beta = residual_norm
        Q[:, 0] = r / beta
        H.zero_()

    return x, {'converged': False, 'iterations': maxiter, 'residual_norm': residual_norm}


