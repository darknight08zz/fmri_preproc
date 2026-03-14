# -*- coding: utf-8 -*-
"""
segment_bias.py
══════════════════════════════════════════════════════════════════
Bias field (INU) estimation and correction

Matches SPM batch:
  Bias Regularisation : 0.0001  (light regularisation)
  Bias FWHM           : 60 mm cutoff
  Save Bias Corrected : Save nothing  (bias field used internally,
                        mT1w.nii saved by segment_save.py)

SPM uses a DCT (Discrete Cosine Transform) basis to model the
smooth, low-frequency bias field.  We reproduce that here.
══════════════════════════════════════════════════════════════════
"""

import sys, io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# SPM batch values — stronger regularisation prevents over-correction
# that collapses WM/CSF contrast (0.0001 was too weak)
BIAS_REG  = 0.01
BIAS_FWHM = 60.0     # mm cutoff — controls how many DCT basis functions kept


class BiasCorrector:
    """
    Estimates and removes the MRI bias field (intensity non-uniformity).

    Algorithm (matches SPM spm_bias_correct approach):
    1. Build a DCT basis limited to spatial frequencies < 1/FWHM
    2. Iteratively estimate bias coefficients using EM-style updates
       (tissue posteriors from GMM guide which voxels are informative)
    3. Reconstruct smooth bias field from DCT coefficients
    4. Return bias-corrected image and the field itself
    """

    def __init__(self,
                 bias_reg:  float = BIAS_REG,
                 bias_fwhm: float = BIAS_FWHM):
        self.bias_reg  = bias_reg
        self.bias_fwhm = bias_fwhm

    # ── DCT basis construction ────────────────────────────────────────

    def _dct_basis(self, n_vox: int, vox_size_mm: float) -> np.ndarray:
        """
        Build 1-D DCT-II basis matrix keeping only low-frequency components
        with wavelength > bias_fwhm mm (i.e. very smooth functions).

        SPM constructs each column as a sampled cosine function:
            B[i, k] = cos(pi * (2*i + 1) * k / (2 * N))   (ortho-normalised)
        This gives smooth spatial basis functions — NOT dct(identity columns),
        which produce random-looking vectors that corrupt tissue contrast.

        Returns (n_vox, n_basis) matrix.
        """
        n_basis = max(1, int(np.floor(n_vox * vox_size_mm / self.bias_fwhm)))
        n_basis = min(n_basis, n_vox)

        i     = np.arange(n_vox, dtype=np.float64)
        basis = np.zeros((n_vox, n_basis), dtype=np.float64)

        for k in range(n_basis):
            col = np.cos(np.pi * (2.0 * i + 1.0) * k / (2.0 * n_vox))
            # Orthonormal scaling: k=0 → 1/sqrt(N),  k>0 → sqrt(2/N)
            col *= (1.0 / np.sqrt(n_vox)) if k == 0 else np.sqrt(2.0 / n_vox)
            basis[:, k] = col

        return basis

    def build_3d_basis(self, shape: tuple, affine: np.ndarray) -> np.ndarray:
        """
        Build the full 3-D DCT basis as a Kronecker product of three 1-D bases.
        Returns (n_voxels, n_coeffs) matrix.
        """
        vox_size = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # mm per voxel

        Bx = self._dct_basis(shape[0], vox_size[0])
        By = self._dct_basis(shape[1], vox_size[1])
        Bz = self._dct_basis(shape[2], vox_size[2])

        # Full basis via Khatri-Rao / Kronecker product
        # B[i,j,k] = Bx[i] ⊗ By[j] ⊗ Bz[k]
        # Vectorised as outer products unrolled
        nx, px = Bx.shape
        ny, py = By.shape
        nz, pz = Bz.shape

        # Precompute Bxy = Bx ⊗ By  (nx*ny, px*py)
        Bxy = np.einsum("ia,jb->ijab", Bx, By).reshape(nx * ny, px * py)
        # Full = Bxy ⊗ Bz  (nx*ny*nz, px*py*pz)
        B3  = np.einsum("ka,lb->klab", Bxy, Bz).reshape(nx * ny * nz, px * py * pz)

        return B3.astype(np.float32)

    # ── Bias estimation ───────────────────────────────────────────────

    def estimate(self,
                 t1w_data:    np.ndarray,
                 posteriors:  np.ndarray,
                 affine:      np.ndarray,
                 verbose:     bool = True,
                 n_iter:      int  = 2) -> tuple:
        """
        Estimate bias field coefficients given tissue posteriors.

        Parameters
        ----------
        t1w_data   : (X,Y,Z) raw T1w
        posteriors : (X,Y,Z,K) tissue posterior probabilities
        affine     : (4,4) voxel-to-world
        n_iter     : EM iterations

        Returns
        -------
        bias_field     : (X,Y,Z) multiplicative bias field
        corrected_data : (X,Y,Z) bias-corrected T1w
        coefficients   : (n_coeffs,) DCT coefficients
        """
        shape  = t1w_data.shape
        n_vox  = int(np.prod(shape))

        if verbose:
            print("  Building DCT bias basis...")
        B = self.build_3d_basis(shape, affine)  # (n_vox, n_coeffs)
        n_coeffs = B.shape[1]

        if verbose:
            print(f"    DCT basis : {n_vox} voxels x {n_coeffs} coefficients")

        # Work in log space: log(observed) = log(true) + log(bias)
        eps       = 1e-6
        log_t1w   = np.log(np.maximum(t1w_data.ravel(), eps))

        # Initialise coefficients to zero (no bias)
        coeffs    = np.zeros(n_coeffs, dtype=np.float64)

        # Regularisation matrix (identity scaled by bias_reg)
        reg_matrix = self.bias_reg * np.eye(n_coeffs)

        for iteration in range(n_iter):
            # Current log bias estimate
            log_bias = B @ coeffs                               # (n_vox,)

            # Log-corrected image (remove current bias estimate)
            log_corr = log_t1w - log_bias                       # (n_vox,)

            # Per-voxel weight = GM + WM posterior only (classes 0 and 1).
            # CSF is excluded because its very low intensity pulls the bias
            # estimator toward over-correction that collapses WM/CSF contrast.
            # Bone/soft/air (classes 3-5) are also excluded — non-brain.
            post_flat = posteriors.reshape(n_vox, -1).astype(np.float64)
            n_class   = post_flat.shape[1]

            # Weight = GM + WM only
            weights  = (post_flat[:, 0] + post_flat[:, 1])
            weights  = np.clip(weights, 0, 1).astype(np.float64)

            # Per-class log-intensity means from the bias-corrected image.
            # These represent the TRUE tissue intensities (bias removed).
            class_means = np.zeros(n_class, dtype=np.float64)
            for k in range(n_class):
                w_k   = weights * post_flat[:, k]
                s_w   = w_k.sum()
                if s_w > 1e-6:
                    class_means[k] = (w_k * log_corr).sum() / s_w

            # Predicted log intensity at each voxel = posterior-weighted class mean
            # This is the "true" signal; bias = observed - predicted
            log_predicted = post_flat @ class_means             # (n_vox,)

            # Residual = log(observed) - log(predicted true signal)
            # = log(bias field)  — this is what we want to fit with DCT
            residual = log_t1w - log_predicted                  # (n_vox,)

            BtW      = B.T * weights[np.newaxis, :]             # (n_coeffs, n_vox)
            BtWB     = BtW @ B + reg_matrix                     # (n_coeffs, n_coeffs)
            BtWr     = BtW @ residual                           # (n_coeffs,)

            try:
                coeffs = np.linalg.solve(BtWB, BtWr)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(BtWB, BtWr, rcond=None)[0]

            if verbose:
                log_bias_tmp = (B @ coeffs).reshape(shape)
                bf_range     = np.exp(log_bias_tmp)
                print(f"    Iter {iteration+1}/{n_iter}  "
                      f"bias range [{bf_range.min():.3f}, {bf_range.max():.3f}]"
                      f"  class means: {np.exp(class_means[:3]).round(1)}")

        # Final bias field
        log_bias_final = (B @ coeffs).reshape(shape)
        bias_field     = np.exp(log_bias_final).astype(np.float32)
        corrected      = (t1w_data / np.maximum(bias_field, eps)).astype(np.float32)

        # Clip corrected to reasonable range
        corrected = np.clip(corrected, 0, t1w_data.max() * 2)

        return bias_field, corrected, coeffs