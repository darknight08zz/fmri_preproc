# -*- coding: utf-8 -*-
"""
segment_gmm.py
══════════════════════════════════════════════════════════════════
Gaussian Mixture Model tissue classification

Matches SPM batch:
  Tissue 1 (GM)           : 1 Gaussian
  Tissue 2 (WM)           : 1 Gaussian
  Tissue 3 (CSF)          : 2 Gaussians
  Tissue 4 (Bone)         : 3 Gaussians
  Tissue 5 (Soft Tissue)  : 4 Gaussians
  Tissue 6 (Air/BG)       : 2 Gaussians

SPM's Unified Segmentation model (Ashburner & Friston 2005):
  - Each tissue class k has N_k Gaussian components
  - TPM provides spatial prior P(class=k | position)
  - Posterior = likelihood × prior / evidence  (Bayes)
  - EM algorithm iterates: E-step (posteriors) → M-step (params)
══════════════════════════════════════════════════════════════════
"""

import sys, io
import numpy as np
from scipy.stats import norm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# Matches your SPM batch exactly
TISSUE_N_GAUSS = [1, 1, 2, 3, 4, 2]   # GM WM CSF Bone SoftTissue Air


class TissueGMM:
    """
    EM-based Gaussian Mixture Model for tissue segmentation.

    Implements SPM's Unified Segmentation approach:
      P(tissue k | voxel i) ∝ P(intensity | tissue k) × P(tissue k | position i)
                                   ↑ GMM likelihood              ↑ TPM prior
    """

    def __init__(self,
                 n_gauss_per_class: list = None,
                 n_iter:            int  = 30,
                 tol:               float = 1e-4,
                 mrf_strength:      float = 1.0):
        """
        Parameters
        ----------
        n_gauss_per_class : list of ints, one per tissue class
                            matches your SPM batch [1,1,2,3,4,2]
        n_iter            : max EM iterations
        tol               : convergence tolerance on log-likelihood
        mrf_strength      : SPM batch MRF Parameter = 1
        """
        self.n_gauss   = n_gauss_per_class or TISSUE_N_GAUSS
        self.n_class   = len(self.n_gauss)
        self.n_iter    = n_iter
        self.tol       = tol
        self.mrf_str   = mrf_strength

        # Total Gaussian components across all classes
        self.n_total   = sum(self.n_gauss)

        # GMM parameters (initialised in fit)
        self.means_    = None   # (n_total,)
        self.vars_     = None   # (n_total,)
        self.weights_  = None   # (n_total,)  mixing coefficients within class
        # Map each component → its tissue class
        self.comp_to_class_ = []
        for k, ng in enumerate(self.n_gauss):
            self.comp_to_class_.extend([k] * ng)

    # ── Initialisation ────────────────────────────────────────────────

    def _init_params(self, intensities: np.ndarray,
                     tpm_flat: np.ndarray) -> None:
        """
        Initialise GMM means using TPM-weighted means of the image intensity.

        SPM anchors each tissue mean to the TPM-weighted average intensity
        for that class.  The old percentile-rank fallback re-ordered classes
        by intensity rather than by tissue index, causing GM↔WM↔CSF swaps.

        T1w intensity order:  CSF(dark) < GM(mid) < WM(bright)
        TPM index order    :  0=GM  1=WM  2=CSF  3=Bone  4=Soft  5=Air

        Fallback anchors (used only when a class has negligible prior mass):
          GM   → 60th percentile of brain intensity
          WM   → 80th percentile  (brightest brain tissue)
          CSF  → 20th percentile  (dark, similar to ventricles)
          Bone → 85th percentile
          Soft → 40th percentile
          Air  →  5th percentile
        These match the known T1w ranking and prevent class swaps.
        """
        # T1w intensity fallback percentiles, one per SPM tissue class index
        FALLBACK_PCT = [60.0, 80.0, 20.0, 85.0, 40.0, 5.0]

        means, variances, weights = [], [], []
        global_std = max(intensities.std(), 1.0)

        for k in range(self.n_class):
            prior_k = tpm_flat[:, k]
            w_sum   = prior_k.sum()

            if w_sum > 1.0:          # enough prior mass → weighted mean
                mu = float((prior_k * intensities).sum() / w_sum)
            else:                    # negligible prior → use known T1w rank
                pct = FALLBACK_PCT[k] if k < len(FALLBACK_PCT) else \
                      (k + 1) / (self.n_class + 1) * 100
                mu = float(np.percentile(intensities, pct))

            # Per-class variance: fraction of global variance
            sigma2 = max((global_std / (self.n_class * 0.8)) ** 2, 1.0)
            ng     = self.n_gauss[k]

            # Spread multiple Gaussians symmetrically around class centre
            spread  = global_std / (self.n_class * 2.0)
            offsets = np.linspace(-spread, spread, ng) if ng > 1 else [0.0]
            for j in range(ng):
                means.append(mu + offsets[j])
                variances.append(sigma2)
                weights.append(1.0 / ng)   # equal mixing within class

        self.means_   = np.array(means,     dtype=np.float64)
        self.vars_    = np.array(variances,  dtype=np.float64)
        self.weights_ = np.array(weights,   dtype=np.float64)

    # ── Gaussian PDF ──────────────────────────────────────────────────

    def _gauss_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all Gaussian components at intensities x.
        Returns (n_vox, n_total).
        """
        eps  = 1e-12
        pdfs = np.zeros((len(x), self.n_total), dtype=np.float64)
        for c in range(self.n_total):
            mu  = self.means_[c]
            sig = np.sqrt(max(self.vars_[c], eps))
            pdfs[:, c] = norm.pdf(x, loc=mu, scale=sig)
        return pdfs + eps

    # ── Class likelihood from component mixture ───────────────────────

    def _class_likelihood(self, pdfs: np.ndarray) -> np.ndarray:
        """
        Aggregate component PDFs into per-class likelihoods.
        P(intensity | class k) = sum_j w_j * N(intensity; mu_j, sig_j)
        Returns (n_vox, n_class).
        """
        n_vox     = pdfs.shape[0]
        lik_class = np.zeros((n_vox, self.n_class), dtype=np.float64)

        comp_idx = 0
        for k, ng in enumerate(self.n_gauss):
            for j in range(ng):
                lik_class[:, k] += self.weights_[comp_idx] * pdfs[:, comp_idx]
                comp_idx += 1

        return lik_class   # (n_vox, n_class)

    # ── EM algorithm ─────────────────────────────────────────────────

    def fit(self,
            t1w_corrected: np.ndarray,
            tpm_data:      np.ndarray,
            verbose:       bool = True) -> np.ndarray:
        """
        Run EM to estimate tissue posteriors.

        Parameters
        ----------
        t1w_corrected : (X,Y,Z) bias-corrected T1w intensities
        tpm_data      : (X,Y,Z,6) tissue prior probabilities

        Returns
        -------
        posteriors : (X,Y,Z,n_class) tissue posterior probabilities
        """
        shape      = t1w_corrected.shape
        n_vox      = int(np.prod(shape))
        x          = t1w_corrected.ravel().astype(np.float64)
        tpm_flat   = tpm_data.reshape(n_vox, self.n_class).astype(np.float64)

        # Clip to valid range, ignore exactly-zero voxels
        x = np.clip(x, 0, None)

        if verbose:
            print("  Initialising GMM parameters from TPM priors...")
        self._init_params(x, tpm_flat)

        prev_ll = -np.inf
        post_comp = None   # (n_vox, n_total)

        for it in range(self.n_iter):

            # ── E-step: compute posteriors ────────────────────────────
            pdfs      = self._gauss_pdf(x)              # (n_vox, n_total)
            lik_class = self._class_likelihood(pdfs)    # (n_vox, n_class)

            # Multiply by TPM spatial priors
            joint     = lik_class * tpm_flat            # (n_vox, n_class)
            evidence  = joint.sum(axis=1, keepdims=True)
            evidence  = np.maximum(evidence, 1e-300)
            class_post = joint / evidence               # (n_vox, n_class) — class posteriors

            # Component posteriors (for M-step)
            comp_idx  = 0
            post_comp = np.zeros((n_vox, self.n_total), dtype=np.float64)
            for k, ng in enumerate(self.n_gauss):
                for j in range(ng):
                    comp_lik  = self.weights_[comp_idx] * pdfs[:, comp_idx]
                    class_lik = lik_class[:, k] + 1e-300
                    post_comp[:, comp_idx] = class_post[:, k] * comp_lik / class_lik
                    comp_idx += 1

            # ── Log-likelihood ────────────────────────────────────────
            ll = np.log(evidence + 1e-300).sum()

            if verbose and (it % 5 == 0 or it == self.n_iter - 1):
                print(f"    EM iter {it+1:2d}/{self.n_iter}  "
                      f"log-likelihood = {ll:.4e}")

            if abs(ll - prev_ll) < self.tol * abs(ll):
                if verbose:
                    print(f"    Converged at iter {it+1}")
                break
            prev_ll = ll

            # ── M-step: update GMM parameters ─────────────────────────
            for c in range(self.n_total):
                r_c       = post_comp[:, c]
                sum_r     = r_c.sum() + 1e-10
                self.means_[c]   = (r_c * x).sum()  / sum_r
                self.vars_[c]    = max(
                    (r_c * (x - self.means_[c]) ** 2).sum() / sum_r,
                    x.var() * 0.01    # lower bound to prevent collapse
                )

            # Update mixing weights within each class
            comp_idx = 0
            for k, ng in enumerate(self.n_gauss):
                class_sum = post_comp[:, comp_idx:comp_idx+ng].sum() + 1e-10
                for j in range(ng):
                    self.weights_[comp_idx] = (
                        post_comp[:, comp_idx].sum() / class_sum
                    )
                    comp_idx += 1

        return class_post.reshape(*shape, self.n_class).astype(np.float32)

    # ── Tissue maps ───────────────────────────────────────────────────

    def get_tissue_maps(self,
                        posteriors: np.ndarray) -> dict:
        """
        Extract the three saved tissue maps from posteriors.

        Returns dict with 'gm', 'wm', 'csf' keys (and raw 'posteriors').
        """
        return {
            "gm"         : posteriors[..., 0],
            "wm"         : posteriors[..., 1],
            "csf"        : posteriors[..., 2],
            "posteriors" : posteriors,
        }