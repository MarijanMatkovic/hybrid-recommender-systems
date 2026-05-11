"""
Smola-Kondor menu of graph-spectral kernels for use as the regulariser
matrix M in Lap-EASE: B* = (G + lambda I + gamma M)^-1 G  (with the
Steck diagonal-zero Lagrangian).

Each kernel is the Tikhonov regulariser corresponding to a different
spectral transform r(.) of the graph Laplacian L = U Lambda U^T:

    Kernel name           r(lambda)               M
    --------------------- ----------------------- ---------------------
    laplacian             lambda                  L                    (current Lap-EASE)
    reg_laplacian         1 + sigma^2 * lambda    I + sigma^2 * L      (isotropic floor)
    heat / diffusion      exp(sigma^2 lambda / 2) expm(sigma^2 L / 2)  (smoother prior)
    p_step_walk           (a - lambda)^-p         (a I - L)^p          (p in {2,3,4})
    inv_cosine            1 / cos(lambda pi/4)    U diag(1/cos)U^T     (high-pass)

Reference:
    Smola & Kondor, "Kernels and Regularization on Graphs", COLT 2003.

The eigendecomposition of L is computed once and cached; switching
kernels is then ~O(n_items^2) (plain matmul rebuild) which is cheap.

Item-item dim is 3.4k on ml-1m, 17.7k on Netflix. We keep the
implementation dense -- callers should not invoke this on Netflix
without check-pointing the eigendecomposition.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


def _project(L, fn):
    """Apply r(.) to the eigenvalues of L: M = U diag(fn(Lambda)) U^T."""
    w, V = eigh(L)
    fw = fn(w)
    return V @ np.diag(fw) @ V.T


def laplacian_kernel(L, **kw):
    """Pure combinatorial Laplacian: M = L (current Lap-EASE penalty)."""
    return np.asarray(L)


def regularized_laplacian(L, sigma2: float = 1.0, **kw):
    """Regularized Laplacian: M = I + sigma^2 * L. Strict superset of
    L2 + Laplacian (you can recover Lap-EASE by taking sigma2 large)."""
    n = L.shape[0]
    return np.eye(n) + sigma2 * np.asarray(L)


def heat_kernel(L, sigma2: float = 1.0, **kw):
    """Diffusion / heat-kernel regulariser:

        r(lambda) = exp(sigma^2 lambda / 2)
        M = U diag(exp(sigma^2 lambda / 2)) U^T

    Exponentially penalises high frequencies. The sigma -> infty limit
    matches GF-CF's ideal low-pass projector.
    """
    return _project(L, lambda w: np.exp(0.5 * sigma2 * w))


def p_step_walk(L, a: float = 2.0, p: int = 2, **kw):
    """p-step random walk regulariser: M = (a I - L)^p.

    For a >= lambda_max(L) this is a high-pass prior. p=2 is the
    smoothest choice; p=3 / p=4 are sharper.

    Note: in the Smola-Kondor framework the p-step walk *kernel* is
    (a - lambda)^-p; the Tikhonov dual M is (a - lambda)^p.
    """
    n = L.shape[0]
    A = a * np.eye(n) - np.asarray(L)
    return np.linalg.matrix_power(A, int(p))


def inverse_cosine(L, lambda_max: float = 2.0, **kw):
    """Inverse-cosine regulariser. Eigenvalues of normalised L lie in
    [0, 2]; we scale to [0, pi/2] so cos(0)=1 (no penalty for the
    zero-frequency component) and cos(pi/2)=0 (infinite penalty for
    highest frequency).
    """
    def fn(w):
        # Clip to [0, lambda_max], then scale so highest eigenvalue
        # lands at pi/2 - epsilon (avoid 1/cos = inf at exactly pi/2).
        w = np.clip(w, 0.0, lambda_max)
        scaled = (w / lambda_max) * (0.5 * np.pi - 1e-3)
        cos_w = np.cos(scaled)
        return 1.0 / np.clip(cos_w, 1e-6, None)
    return _project(L, fn)


# Single-entry registry. Each entry: (callable, default kwargs).
KERNELS = {
    'laplacian':      laplacian_kernel,
    'reg_laplacian':  regularized_laplacian,
    'heat':           heat_kernel,
    'p_step_2':       lambda L, **kw: p_step_walk(L, p=2, **kw),
    'p_step_3':       lambda L, **kw: p_step_walk(L, p=3, **kw),
    'inv_cosine':     inverse_cosine,
}


def build_kernel(name: str, L, **kw):
    """Build M for kernel ``name`` from Laplacian L and optional kwargs."""
    if name not in KERNELS:
        raise ValueError(
            f"Unknown spectral kernel '{name}'. "
            f"Available: {sorted(KERNELS)}")
    return np.ascontiguousarray(KERNELS[name](L, **kw))
