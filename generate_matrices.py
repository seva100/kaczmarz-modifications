import numpy as np
import scipy as sp
import scipy.sparse


def over_determined(n, m, ampl=50):
    return np.random.randn(n, m) * ampl - ampl / 2


def hilbert(n):
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    return 1.0 / (i_grid + j_grid + 1)


def rand_for_given_cond_number(n, m, cond=1, rank='full'):
    """Samples random matrix of size (n, m) with given condition number `cond` (in 2-norm)
    and given rank `rank` ('full' or number)."""

    full_rank = False
    while not full_rank:
        print('not full rank')
        sample_mx = np.random.rand(n, m)
        U, sing_vals, V = np.linalg.svd(sample_mx)
        if np.all(~np.isclose(sing_vals, 0.0)):
            full_rank = True

    if rank == 'full':
        rank = min(n, m)
    sing_vals = np.linspace(1.0, cond, rank)[::-1]
    print('sing_vals:', sing_vals)
    if rank < min(n, m):
        sing_vals = np.concatenate([sing_vals, np.zeros(min(n, m) - rank)])
    print('sing_vals:', sing_vals)
    Sigma = np.diag(sing_vals)
    if Sigma.shape[0] < n:
        Sigma = np.vstack([Sigma, np.zeros((n - Sigma.shape[0], m))])
    if Sigma.shape[1] < m:
        Sigma = np.hstack([Sigma, np.zeros((n, m - Sigma.shape[1]))])
    print('Sigma:', Sigma)
    mx = U @ Sigma @ V
    return mx


def band_mx(n, m, half_band_size=1, ampl=50, return_sparse=True):
    """We are considering that band is always of size (2 * half_band_size + 1) (so it is always odd).
    """
    diag_vals = np.random.randn(2 * half_band_size + 1) * ampl - ampl / 2
    print(diag_vals)
    diag_data = np.repeat(diag_vals[:, np.newaxis], min(n, m), axis=1)
    print(diag_data)
    print(np.arange(-half_band_size, half_band_size + 1, dtype=np.int64))
    mx = sp.sparse.spdiags(diag_data, np.arange(-half_band_size, half_band_size + 1, dtype=np.int64), n, m)
    if return_sparse:
        return mx
    return mx.toarray()


def band_mx_with_given_cond(n, m, half_band_size=1, init_ampl=50, cond=1, cond_tol=1e-2):
    """We are considering that band is always of size (2 * half_band_size + 1) (so it is always odd).
    """
    diag_vals = np.random.randn(2 * half_band_size + 1) * init_ampl - init_ampl / 2
    diag_data = np.repeat(diag_vals[:, np.newaxis], min(n, m), axis=1)
    mx = sp.sparse.spdiags(diag_data, np.arange(-half_band_size, half_band_size + 1, dtype=np.int64), n, m).toarray()
    print(mx)
    
    mx_new = mx.copy()
    
    left_fr, right_fr = 0, 1e+7     # be careful with this number! wrong value can produce:
                                    # 1. Floating points overflows (need smaller)
                                    # 2. Unreachable condition number (need bigger)
    factor = (left_fr + right_fr) / 2
    factor_prev = 1
    cur_cond = -1
    while np.abs(cur_cond - cond) > cond_tol:
        mx_new[np.arange(min(n, m) - half_band_size), np.arange(half_band_size, min(n, m))] *= (factor / factor_prev)
        cur_cond = np.linalg.cond(mx_new)
        if cur_cond < cond:
            left_fr = factor
        else:
            right_fr = factor
        factor_prev = factor
        factor = (left_fr + right_fr) / 2
    
    return mx_new
