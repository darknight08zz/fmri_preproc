# -*- coding: utf-8 -*-
"""
segment_load.py
══════════════════════════════════════════════════════════════════
Load and validate T1w structural image + SPM TPM.nii

SPM batch inputs matched:
  Volumes              : T1w.nii
  Tissue Prob Maps     : TPM.nii  (6 volumes, one per tissue class)
══════════════════════════════════════════════════════════════════
"""

import sys, io
import numpy as np
import nibabel as nib
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# SPM tissue class definitions matching your batch exactly
TISSUE_CLASSES = [
    {"name": "Gray Matter",       "tpm_idx": 0, "n_gauss": 1, "save_native": True},
    {"name": "White Matter",      "tpm_idx": 1, "n_gauss": 1, "save_native": True},
    {"name": "CSF",               "tpm_idx": 2, "n_gauss": 2, "save_native": True},
    {"name": "Bone",              "tpm_idx": 3, "n_gauss": 3, "save_native": False},
    {"name": "Soft Tissue",       "tpm_idx": 4, "n_gauss": 4, "save_native": False},
    {"name": "Air/Background",    "tpm_idx": 5, "n_gauss": 2, "save_native": False},
]


class SegmentLoader:
    """
    Loads T1w image and TPM.nii, returns data arrays and metadata
    needed by all downstream segmentation modules.
    """

    def __init__(self, t1w_path: str, tpm_path: str):
        self.t1w_path = Path(t1w_path)
        self.tpm_path = Path(tpm_path)

        if not self.t1w_path.exists():
            raise FileNotFoundError(f"T1w not found: {self.t1w_path}")
        if not self.tpm_path.exists():
            raise FileNotFoundError(
                f"TPM.nii not found: {self.tpm_path}\n"
                f"Point to <SPM_dir>/tpm/TPM.nii on your system."
            )

    def load(self, verbose: bool = True) -> dict:
        """
        Load T1w and TPM, return a context dict shared across all modules.

        Returns
        -------
        ctx : dict with keys
            t1w_data     : (X,Y,Z) float32  — raw T1w intensities
            t1w_affine   : (4,4)             — voxel-to-world mm
            t1w_header   : NIfTI header
            t1w_shape    : (X,Y,Z)
            tpm_data     : (X,Y,Z,6) float32 — tissue prior probabilities
            tpm_affine   : (4,4)
            tpm_shape    : (X,Y,Z)
            n_voxels     : int
            tissue_classes : list of dicts
            t1w_path     : Path
            tpm_path     : Path
        """
        if verbose:
            print("  Loading T1w structural image...")

        t1w_img    = nib.load(str(self.t1w_path))
        t1w_data   = np.asarray(t1w_img.get_fdata(dtype="float32"))
        t1w_affine = t1w_img.affine.copy()
        t1w_header = t1w_img.header

        # Squeeze out any extra dimensions (some DICOM converters add a 4th)
        if t1w_data.ndim == 4 and t1w_data.shape[3] == 1:
            t1w_data = t1w_data[..., 0]
        if t1w_data.ndim != 3:
            raise ValueError(
                f"T1w must be 3D. Got shape: {t1w_data.shape}"
            )

        if verbose:
            print(f"    Shape  : {t1w_data.shape}")
            print(f"    Affine : vox size = "
                  f"{np.sqrt((t1w_affine[:3,:3]**2).sum(0)).round(3)} mm")
            print(f"    Range  : [{t1w_data.min():.1f}, {t1w_data.max():.1f}]")

        if verbose:
            print("  Loading TPM.nii...")

        tpm_img  = nib.load(str(self.tpm_path))
        tpm_data = np.asarray(tpm_img.get_fdata(dtype="float32"))

        # TPM.nii must be 4D with 6 volumes
        if tpm_data.ndim != 4 or tpm_data.shape[3] < 6:
            raise ValueError(
                f"TPM.nii must be 4D with >=6 volumes. Got: {tpm_data.shape}"
            )
        tpm_data = tpm_data[..., :6]   # keep only the 6 standard classes

        if verbose:
            print(f"    TPM shape : {tpm_data.shape}  (X Y Z 6-classes)")

        # Resample TPM to T1w voxel grid if shapes differ
        tpm_on_t1w = self._resample_tpm_to_t1w(
            tpm_data, tpm_img.affine, t1w_data.shape, t1w_affine, verbose
        )

        # Clamp & normalise TPM so probabilities sum to <=1 per voxel
        tpm_on_t1w = np.clip(tpm_on_t1w, 0.0, 1.0)
        tpm_sum    = tpm_on_t1w.sum(axis=3, keepdims=True)
        tpm_sum    = np.where(tpm_sum < 1e-6, 1.0, tpm_sum)
        tpm_on_t1w = tpm_on_t1w / tpm_sum

        # Brain mask: GM+WM+CSF prior sum > threshold
        # Prevents skull/neck/fat from being misclassified as GM
        t1w_data, brain_mask = self._apply_brain_mask(
            t1w_data, tpm_on_t1w, verbose
        )

        return {
            "t1w_data"      : t1w_data,
            "brain_mask"    : brain_mask,
            "t1w_affine"    : t1w_affine,
            "t1w_header"    : t1w_header,
            "t1w_shape"     : t1w_data.shape,
            "tpm_data"      : tpm_on_t1w,
            "tpm_affine"    : t1w_affine,   # now on T1w grid
            "n_voxels"      : int(np.prod(t1w_data.shape)),
            "tissue_classes": TISSUE_CLASSES,
            "t1w_path"      : self.t1w_path,
            "tpm_path"      : self.tpm_path,
        }

    # ── Brain masking ─────────────────────────────────────────────────

    def _apply_brain_mask(self,
                          t1w_data:   np.ndarray,
                          tpm_on_t1w: np.ndarray,
                          verbose:    bool) -> tuple:
        """
        Build a brain mask from TPM priors and zero out non-brain voxels.

        Strategy (matches SPM's implicit masking via tissue priors):
          1. Brain prior = sum of GM + WM + CSF probability maps (classes 0-2)
          2. Threshold at 0.2 → initial binary mask
          3. Morphological closing (fill small holes) then keep largest component
          4. Zero t1w_data outside mask so skull/neck never enters GMM

        Returns
        -------
        masked_t1w  : (X,Y,Z) float32 — t1w with non-brain zeroed
        brain_mask  : (X,Y,Z) bool    — True = brain voxel
        """
        from scipy.ndimage import (binary_closing, binary_fill_holes,
                                   label as ndlabel)

        # Step 1: brain probability = GM + WM + CSF
        brain_prob = tpm_on_t1w[..., 0] + tpm_on_t1w[..., 1] + tpm_on_t1w[..., 2]

        # Step 2: threshold
        brain_mask = brain_prob > 0.2

        # Step 3: morphological closing (3-iteration) to fill sulcal gaps,
        #         then fill holes slice-by-slice, then keep largest component
        struct     = np.ones((3, 3, 3), dtype=bool)
        brain_mask = binary_closing(brain_mask, structure=struct, iterations=3)

        # Fill holes in each axial slice independently (faster, robust)
        for z in range(brain_mask.shape[2]):
            brain_mask[:, :, z] = binary_fill_holes(brain_mask[:, :, z])

        # Keep only the largest connected component (removes stray blobs)
        labelled, n_comp = ndlabel(brain_mask, structure=struct)
        if n_comp > 1:
            sizes          = np.bincount(labelled.ravel())
            sizes[0]       = 0          # ignore background
            brain_mask     = labelled == sizes.argmax()

        brain_mask = brain_mask.astype(bool)

        # Step 4: apply mask
        masked_t1w = t1w_data.copy()
        masked_t1w[~brain_mask] = 0.0

        if verbose:
            n_brain = brain_mask.sum()
            pct     = 100.0 * n_brain / brain_mask.size
            print(f"    Brain mask : {n_brain:,} voxels  ({pct:.1f}% of FOV)")

        return masked_t1w.astype(np.float32), brain_mask

    # ── TPM resampling ────────────────────────────────────────────────

    def _resample_tpm_to_t1w(self,
                              tpm_data:   np.ndarray,
                              tpm_affine: np.ndarray,
                              t1w_shape:  tuple,
                              t1w_affine: np.ndarray,
                              verbose:    bool) -> np.ndarray:
        """
        Resample TPM volumes onto the T1w voxel grid using trilinear
        interpolation. SPM does this internally before segmentation.
        """
        from scipy.ndimage import map_coordinates

        if tpm_data.shape[:3] == t1w_shape:
            if verbose:
                print("    TPM already on T1w grid — no resampling needed")
            return tpm_data

        if verbose:
            print(f"    Resampling TPM {tpm_data.shape[:3]} → {t1w_shape}...")

        # Build T1w voxel grid in world mm
        x, y, z = [np.arange(s) for s in t1w_shape]
        xi, yi, zi = np.meshgrid(x, y, z, indexing="ij")
        t1w_vox_hom = np.vstack([xi.ravel(), yi.ravel(), zi.ravel(),
                                  np.ones(xi.size)])         # (4, N)
        world_mm    = t1w_affine @ t1w_vox_hom               # (4, N)

        # Convert world mm → TPM voxel coords
        tpm_aff_inv = np.linalg.inv(tpm_affine)
        tpm_vox     = (tpm_aff_inv @ world_mm)[:3]           # (3, N)

        n_classes  = tpm_data.shape[3]
        resampled  = np.zeros((*t1w_shape, n_classes), dtype=np.float32)

        for k in range(n_classes):
            resampled[..., k] = map_coordinates(
                tpm_data[..., k], tpm_vox,
                order=1, mode="nearest", prefilter=False
            ).reshape(t1w_shape)

        return resampled