import numpy as np


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
    sing_vals = np.linspace(1.0, cond, rank)
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
