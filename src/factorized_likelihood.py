"""
Stage 5 & 7 (todo.txt Sec. 7, 9): exploratory low-rank factorization and the
Woodbury-identity likelihood for C = D + B B^T, with D diagonal.

D is assumed diagonal throughout this module (the statistical covariance in
this project's C = D + U W U^T model). B already absorbs W, i.e.
B = U_K @ diag(sqrt(w_1..w_K)) -- see build_low_rank_from_eigh() for the
exploratory eigendecomposition version, or pass a physically-motivated B
(e.g. scaled systematic Delta_mu vectors) directly.
"""

import logging

import numpy as np
import scipy.linalg

log = logging.getLogger(__name__)


def build_low_rank_from_eigh(eigenvalues, eigenvectors, k):
    """
    Exploratory numerical factorization (todo.txt Stage 5):
    C_syst ~= U_K Lambda_K U_K^T, B = U_K Lambda_K^{1/2}.

    eigenvalues/eigenvectors are expected in the ascending order returned by
    scipy.linalg.eigh; the top-k (largest) modes are used.

    Negative or non-positive eigenvalues among the requested top-k are
    reported, not silently dropped or clipped -- the caller decides whether
    to reduce k.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)
    n = len(eigenvalues)
    if k > n:
        raise ValueError(f"k={k} exceeds matrix dimension {n}")

    top_idx = np.argsort(eigenvalues)[::-1][:k]
    top_vals = eigenvalues[top_idx]
    top_vecs = eigenvectors[:, top_idx]

    n_nonpositive = int(np.sum(top_vals <= 0))
    if n_nonpositive > 0:
        log.warning(
            f"build_low_rank_from_eigh(k={k}): {n_nonpositive} of the requested "
            f"top-{k} eigenvalues are <= 0 (min={top_vals.min():.3e}); "
            f"these modes will produce NaN in sqrt() unless k is reduced."
        )

    B = top_vecs * np.sqrt(np.clip(top_vals, 0, None))[None, :]
    return B, top_vals, n_nonpositive


def reconstruction_error(C, C_hat):
    """
    todo.txt Stage 5: E_F = ||C - C_hat||_F / ||C||_F,
    E_max = max|C - C_hat| / max|C|.
    """
    C = np.asarray(C, dtype=np.float64)
    C_hat = np.asarray(C_hat, dtype=np.float64)
    diff = C - C_hat

    e_frob = np.linalg.norm(diff, ord="fro") / np.linalg.norm(C, ord="fro")
    e_max = np.max(np.abs(diff)) / np.max(np.abs(C))
    return float(e_frob), float(e_max)


class FactorizedCovariance:
    """
    C = D + B B^T, D diagonal (length-N array), B is N x K.

    Implements the Woodbury identity and matrix determinant lemma so that
    C^-1 and log|C| are never computed via a dense N x N inverse:

        C^-1 = D^-1 - D^-1 B (I + B^T D^-1 B)^-1 B^T D^-1
        log|C| = log|D| + log|I + B^T D^-1 B|

    All K x K and N x K quantities that do not depend on the residual are
    cached at construction time.
    """

    def __init__(self, diag, B):
        self.diag = np.asarray(diag, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        n, k = self.B.shape
        if self.diag.shape[0] != n:
            raise ValueError(f"diag length {self.diag.shape[0]} != B rows {n}")
        if np.any(self.diag <= 0):
            raise ValueError("D must be strictly positive (diagonal statistical variances)")

        self.n = n
        self.k = k

        self.d_inv = 1.0 / self.diag
        self.d_inv_B = self.d_inv[:, None] * self.B  # D^-1 B

        M = np.eye(k) + self.B.T @ self.d_inv_B
        self.M_chol = scipy.linalg.cholesky(M, lower=True)

        self.logdet_D = float(np.sum(np.log(self.diag)))
        self.logdet_M = float(2.0 * np.sum(np.log(np.diag(self.M_chol))))

    def quadratic_form(self, residual):
        """r^T C^-1 r via Woodbury, without forming C^-1 or C."""
        r = np.asarray(residual, dtype=np.float64)
        y = self.d_inv * r
        q = self.B.T @ y
        m_inv_q = scipy.linalg.cho_solve((self.M_chol, True), q)
        return float(r @ y - q @ m_inv_q)

    def apply_inverse(self, vector):
        """C^-1 v via Woodbury."""
        v = np.asarray(vector, dtype=np.float64)
        y = self.d_inv * v
        q = self.B.T @ y
        m_inv_q = scipy.linalg.cho_solve((self.M_chol, True), q)
        return y - self.d_inv_B @ m_inv_q

    def logdet(self):
        return self.logdet_D + self.logdet_M

    def loglike(self, residual, include_norm=True):
        chi2 = self.quadratic_form(residual)
        ld = self.logdet()
        neg2loglike = chi2 + ld
        if include_norm:
            neg2loglike += self.n * np.log(2 * np.pi)
        return chi2, ld, -0.5 * neg2loglike

    def to_dense(self):
        """Diagnostic only: materialize C = D + B B^T as a dense array."""
        return np.diag(self.diag) + self.B @ self.B.T
