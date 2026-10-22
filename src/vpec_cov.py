"""
Peculiar-velocity covariance for SN samples, in the factorized form
C = D + B B^T used by factorized_likelihood.FactorizedCovariance.

Steps provided here:
  1. read RA, DEC, zCMB from a FITRES table (read_fitres)
  2. call FLIP (Ravoux et al., arXiv:2501.16852) for the velocity-velocity
     covariance P(RA, DEC, zCMB) in (km/s)^2 (flip_vv_covariance)
  3. convert velocity to distance-modulus shifts (vpec_to_dmu)
  4. build B either from random realizations (monte_carlo_B, the method
     proposed by R. Kessler) or from the top-K eigenmodes (eigen_B)

FLIP is expected in third_party/flip and is imported from there, without
installing it.
"""

import logging
import os
import sys

import numpy as np
import scipy.linalg
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factorized_likelihood import FactorizedCovariance  # noqa: E402

log = logging.getLogger(__name__)

C_LIGHT = 299792.458  # km/s

_HERE = os.path.dirname(os.path.abspath(__file__))
FLIP_DIR = os.path.join(_HERE, "..", "third_party", "flip")


def read_fitres(path, columns=("RA", "DEC", "zCMB")):
    """Read selected columns of a SNANA FITRES text table into a dict of arrays."""
    names = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    if names is None or not rows:
        raise ValueError(f"{path}: no VARNAMES/SN lines found")
    idx = {name: names.index(name) for name in columns}
    return {name: np.array([float(r[i]) for r in rows]) for name, i in idx.items()}


def comoving_distance_mpc_h(zcmb, om=0.3, n_grid=4000):
    """Comoving distance in Mpc/h for flat LCDM, the distance unit FLIP expects."""
    zcmb = np.asarray(zcmb, dtype=np.float64)
    zg = np.linspace(0.0, zcmb.max() * 1.001, n_grid)
    ez = np.sqrt(om * (1 + zg) ** 3 + 1 - om)
    integrand = 1.0 / ez
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(zg))])
    return (C_LIGHT / 100.0) * np.interp(zcmb, zg, cum)


def _import_flip(flip_dir=FLIP_DIR):
    os.environ.setdefault("FLIP_USE_JAX", "0")
    if flip_dir not in sys.path:
        sys.path.insert(0, flip_dir)
    from flip.covariance import covariance  # noqa: E402
    from flip.data import load_data_test  # noqa: E402

    return covariance, load_data_test


def flip_vv_covariance(ra_deg, dec_deg, zcmb, power_spectrum_dict=None, om=0.3):
    """
    Velocity-velocity covariance per unit fs8^2, in (km/s)^2.

    Returns P with C_vv(fs8, sigv) = fs8^2 * P + sigv^2 * I. Because the
    nuisance parameters only rescale P or add to the diagonal, P is computed
    once and the Woodbury factors are reused for every MCMC step.

    power_spectrum_dict defaults to the power spectrum bundled with FLIP
    (a placeholder until the FLIP team supplies the production file).
    """
    covariance, load_data_test = _import_flip()
    if power_spectrum_dict is None:
        power_spectrum_dict = load_data_test.load_power_spectrum_dict()

    r = comoving_distance_mpc_h(zcmb, om=om)
    coords = np.array([np.asarray(ra_deg), np.asarray(dec_deg), r])
    n = coords.shape[1]

    cov = covariance.CovMatrix.init_from_flip(
        "adamsblake17plane",
        "velocity",
        power_spectrum_dict,
        coordinates_density=None,
        coordinates_velocity=coords,
        size_batch=50_000,
        number_worker=1,
    )
    cov.init_compute_covariance_sum()
    P = np.asarray(cov.compute_covariance_sum({"fs8": 1.0, "sigv": 0.0}, np.zeros(n)))
    return 0.5 * (P + P.T)


def vpec_to_dmu(zcmb):
    """
    Factor f(z) such that dmu = f(z) * vpec for vpec in km/s (low-z, flat
    approximation of Davis et al. 2011):  f = 5/ln10 * (1+z)/(z(1+z/2)) / c.
    """
    z = np.asarray(zcmb, dtype=np.float64)
    return (5.0 / np.log(10.0)) * (1.0 + z) / (z * (1.0 + 0.5 * z)) / C_LIGHT


def eigen_spectrum(P):
    """Eigenvalues of P in descending order (negatives from round-off kept)."""
    return np.linalg.eigvalsh(P)[::-1]


def n_modes_for_variance(eigenvalues, fraction):
    """Smallest K whose top-K eigenvalues hold the given fraction of the trace."""
    pos = np.clip(eigenvalues, 0, None)
    cum = np.cumsum(pos) / pos.sum()
    return int(np.searchsorted(cum, fraction) + 1)


def eigh_sorted(C):
    """Eigendecomposition of C (ascending), computed once and passed to the B builders."""
    return np.linalg.eigh(C)


def eigen_B(C, k, eig=None):
    """Deterministic low-rank factor from the top-k eigenmodes, C ~ B B^T."""
    w, v = eig if eig is not None else np.linalg.eigh(C)
    top = np.argsort(w)[::-1][:k]
    return v[:, top] * np.sqrt(np.clip(w[top], 0, None))[None, :]


def monte_carlo_B(C, k, rng, eig=None):
    """
    Random-realization factor: draw k vectors x_j ~ N(0, C) and set
    B = [x_1 ... x_k] / sqrt(k), so that E[B B^T] = C and every column is a
    systematic shift vector like a calibration term. Works for any rank of C
    because the draws use the full eigendecomposition.
    """
    w, v = eig if eig is not None else np.linalg.eigh(C)
    root = v * np.sqrt(np.clip(w, 0, None))[None, :]
    x = root @ rng.standard_normal((C.shape[0], k))
    return x / np.sqrt(k)


def remainder_diag(C, B):
    """
    Diagonal of the part of C that B B^T misses, clipped at zero. Adding it to
    D restores the per-SN variance and removes most of the log|C| bias from the
    dropped small-eigenvalue modes. For eigen_B it is exactly non-negative; for
    monte_carlo_B it is noisy and the clip makes the correction slightly positive.
    """
    return np.clip(np.diag(C) - np.sum(B**2, axis=1), 0.0, None)


def randomized_B(C, k, rng, n_power=1):
    """
    Randomized low-rank factor (Halko, Martinsson and Tropp 2011): sketch C with
    k Gaussian vectors, orthonormalize, and diagonalize the projected k x k
    matrix. Needs only about (2 n_power + 2) products C @ (N x k) instead of a
    full eigendecomposition, and approaches eigen_B as k grows.
    """
    n = C.shape[0]
    Q, _ = np.linalg.qr(C @ rng.standard_normal((n, k)))
    for _ in range(n_power):
        Q, _ = np.linalg.qr(C @ Q)
    w, v = np.linalg.eigh(Q.T @ C @ Q)
    return (Q @ v) * np.sqrt(np.clip(w, 0, None))[None, :]


def m2lnl_dense(f, r, d, P):
    """-2 ln L (without the 2 pi term) for C = D + f^2 P, by a dense Cholesky."""
    L = scipy.linalg.cholesky(np.diag(d) + f**2 * P, lower=True)
    y = scipy.linalg.solve_triangular(L, r, lower=True)
    return float(y @ y + 2.0 * np.sum(np.log(np.diag(L))))


def m2lnl_woodbury(f, r, d, B1, rem1=None):
    """
    Same quantity from the factorized form C = D + f^2 rem1 + (f B1)(f B1)^T,
    where B1 and rem1 are the fs8 = 1 columns and remainder diagonal.
    """
    rem = 0.0 if rem1 is None else f**2 * rem1
    fc = FactorizedCovariance(d + rem, f * B1)
    return fc.quadratic_form(r) + fc.logdet()


def fit_fs8(fun, lo=0.02, hi=1.5):
    """Minimize a -2 ln L function of fs8 on [lo, hi]."""
    return minimize_scalar(fun, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4}).x


def sigma_from_curvature(fun, f0, h=0.02):
    """1-sigma error from the curvature of -2 ln L at its minimum f0."""
    f2 = (fun(f0 + h) - 2.0 * fun(f0) + fun(f0 - h)) / h**2
    return float(np.sqrt(2.0 / f2))
