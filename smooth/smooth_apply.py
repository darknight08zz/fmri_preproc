# -*- coding: utf-8 -*-
"""
smooth_apply.py
══════════════════════════════════════════════════════════════════
SPM Smooth — Gaussian spatial smoothing

Matches SPM batch exactly:
  FWHM            : [6 6 6] mm
  Data Type       : SAME  (float32 preserved)
  Implicit masking: No
  Prefix          : s

Algorithm (matches SPM spm_smooth.m):
  1. Convert FWHM mm → sigma in voxels per axis
         sigma_vox[d] = FWHM_mm[d] / (2 * sqrt(2*ln(2)) * vox_size_mm[d])
  2. Apply separable 1-D Gaussian convolution along each axis:
         vol_x  = gaussian_filter1d(vol,  sigma_vox[0], axis=0)
         vol_xy = gaussian_filter1d(vol_x, sigma_vox[1], axis=1)
         vol_smooth = gaussian_filter1d(vol_xy, sigma_vox[2], axis=2)
     (Separable = mathematically identical to 3-D convolution, ~100x faster)
  3. No implicit masking — zeros included in smoothing (SPM batch setting)
  4. Data type SAME — output kept as float32
  5. Save with prefix s
══════════════════════════════════════════════════════════════════
"""

import sys, io
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# SPM batch values
FWHM_MM   = [6.0, 6.0, 6.0]   # mm per axis
PREFIX    = "s"

# Conversion constant: FWHM = 2 * sqrt(2 * ln(2)) * sigma
FWHM2SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # ≈ 0.4247


class Smoother:
    """
    SPM-matched Gaussian spatial smoother.

    Applies separable 1-D Gaussian convolution along each spatial axis,
    which is mathematically identical to 3-D Gaussian convolution.
    """

    def __init__(self, fwhm_mm: list = None):
        """
        Parameters
        ----------
        fwhm_mm : [fx, fy, fz] FWHM in mm per axis. Default [6, 6, 6].
        """
        self.fwhm_mm = fwhm_mm or FWHM_MM

    def _sigma_voxels(self, affine: np.ndarray) -> np.ndarray:
        """
        Convert FWHM mm → sigma in voxels for each axis.

        sigma_vox[d] = FWHM_mm[d] * FWHM2SIGMA / vox_size_mm[d]

        Parameters
        ----------
        affine : (4,4) voxel-to-world affine of the image

        Returns
        -------
        sigma : (3,) array of sigma values in voxels
        """
        vox_size = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # mm per voxel
        sigma    = np.array(self.fwhm_mm) * FWHM2SIGMA / vox_size
        return sigma

    def smooth_volume(self,
                      vol:   np.ndarray,
                      sigma: np.ndarray) -> np.ndarray:
        """
        Smooth a single 3-D volume with separable Gaussian kernel.

        Parameters
        ----------
        vol   : (X,Y,Z) float32 volume
        sigma : (3,) sigma in voxels per axis

        Returns
        -------
        smoothed : (X,Y,Z) float32
        """
        out = vol.astype(np.float64)

        for axis, s in enumerate(sigma):
            if s > 0.0:
                out = gaussian_filter1d(
                    out,
                    sigma=s,
                    axis=axis,
                    mode="reflect",   # matches SPM boundary handling
                    truncate=4.0,     # kernel truncated at 4σ (SPM default)
                )

        return out.astype(np.float32)

    def run(self,
            func_path: str,
            verbose:   bool = True) -> str:
        """
        Smooth a 4D NIfTI volume-by-volume and save with prefix s.

        Parameters
        ----------
        func_path : path to wrarfunc_4D.nii

        Returns
        -------
        out_path : path to swrarfunc_4D.nii
        """
        import time
        t0 = time.time()

        func_path = Path(func_path)

        # ── Load ──────────────────────────────────────────────────────
        if verbose:
            print("  Loading functional image...")
        img    = nib.load(str(func_path))
        data   = img.get_fdata(dtype="float32")
        affine = img.affine.copy()
        header = img.header

        if data.ndim == 3:
            data = data[..., np.newaxis]
        n_vols = data.shape[3]

        vox_size = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).round(4)

        # ── Compute sigma ─────────────────────────────────────────────
        sigma = self._sigma_voxels(affine)

        if verbose:
            print(f"    Shape    : {data.shape}")
            print(f"    Vox size : {vox_size} mm")
            print(f"    FWHM     : {self.fwhm_mm} mm")
            print(f"    Sigma    : {sigma.round(4)} voxels")

        # ── Smooth each volume ────────────────────────────────────────
        if verbose:
            print(f"  Smoothing {n_vols} volumes...")

        out_data = np.zeros_like(data, dtype=np.float32)

        for v in range(n_vols):
            if verbose and (v % 50 == 0 or v == n_vols - 1):
                print(f"    Volume {v+1:4d}/{n_vols}")
            out_data[..., v] = self.smooth_volume(data[..., v], sigma)

        # ── Save ──────────────────────────────────────────────────────
        out_path = func_path.parent / (PREFIX + func_path.name)

        out_img = nib.Nifti1Image(out_data, affine, header)
        out_img.set_sform(affine, code=1)
        out_img.set_qform(affine, code=1)

        # Keep same data type (SAME setting)
        out_img.set_data_dtype(np.float32)

        nib.save(out_img, str(out_path))

        elapsed = time.time() - t0

        if verbose:
            print(f"\n  Output : {out_path.name}")
            print(f"  Shape  : {out_data.shape}")
            print(f"  Range  : [{out_data.min():.2f}, {out_data.max():.2f}]")
            print(f"  Time   : {elapsed:.1f}s")

        return str(out_path)