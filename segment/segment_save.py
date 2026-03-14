# -*- coding: utf-8 -*-
"""
segment_save.py
══════════════════════════════════════════════════════════════════
Save all segmentation outputs in SPM naming convention

Output files:
  c1<name>.nii   — Gray Matter probability map
  c2<name>.nii   — White Matter probability map
  c3<name>.nii   — CSF probability map
  y_<name>.nii   — Forward deformation field  (subject → MNI)
  iy_<name>.nii  — Inverse deformation field  (MNI → subject)
  m<name>.nii    — Bias-corrected T1w
══════════════════════════════════════════════════════════════════
"""

import sys, io
import numpy as np
import nibabel as nib
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")


class SegmentSaver:
    """Saves all segmentation outputs with SPM-style filenames."""

    def __init__(self, t1w_path: str):
        p           = Path(t1w_path)
        # Strip .nii or .nii.gz to get the bare stem
        stem        = p.name
        for ext in [".nii.gz", ".nii"]:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        self.stem   = stem
        self.outdir = p.parent
        self.suffix = ".nii"

    # ── helpers ───────────────────────────────────────────────────────

    def _path(self, prefix: str) -> Path:
        return self.outdir / f"{prefix}{self.stem}{self.suffix}"

    def _save_3d(self,
                 data:   np.ndarray,
                 affine: np.ndarray,
                 header,
                 path:   Path,
                 dtype=np.float32) -> str:
        data  = data.astype(dtype)
        img   = nib.Nifti1Image(data, affine, header)
        img.set_sform(affine, code=1)
        img.set_qform(affine, code=1)
        nib.save(img, str(path))
        return str(path)

    # ── tissue maps ───────────────────────────────────────────────────

    def save_tissue_maps(self,
                         posteriors: np.ndarray,
                         t1w_affine: np.ndarray,
                         t1w_header,
                         verbose:    bool = True) -> list:
        """
        Save c1, c2, c3 (GM, WM, CSF) probability maps.

        posteriors : (X,Y,Z,>=3)
        Returns list of saved paths.
        """
        saved = []
        labels = [("c1", "Gray Matter"),
                  ("c2", "White Matter"),
                  ("c3", "CSF")]

        for i, (prefix, name) in enumerate(labels):
            p    = self._path(prefix)
            path = self._save_3d(posteriors[..., i], t1w_affine, t1w_header, p)
            saved.append(path)
            if verbose:
                vox_count = (posteriors[..., i] > 0.5).sum()
                print(f"    {prefix}{self.stem}.nii  [{name}]  "
                      f"{vox_count:,} dominant voxels")

        return saved

    # ── bias-corrected T1w ────────────────────────────────────────────

    def save_bias_corrected(self,
                             corrected:  np.ndarray,
                             t1w_affine: np.ndarray,
                             t1w_header,
                             verbose:    bool = True) -> str:
        p    = self._path("m")
        path = self._save_3d(corrected, t1w_affine, t1w_header, p)
        if verbose:
            print(f"    m{self.stem}.nii  [Bias-corrected T1w]")
        return path

    # ── deformation fields ────────────────────────────────────────────

    def save_forward_field(self,
                            y_field:    np.ndarray,
                            t1w_affine: np.ndarray,
                            t1w_header,
                            verbose:    bool = True) -> str:
        """
        Save y_<stem>.nii — forward deformation field.

        SPM stores deformation fields as (X,Y,Z,1,3) NIfTI with intent code
        NIFTI_INTENT_VECTOR (1007).  The three volumes are the x,y,z MNI
        world coordinates (mm) for each subject voxel.
        """
        p = self._path("y_")

        # Ensure shape is (X,Y,Z,1,3)
        if y_field.ndim == 4 and y_field.shape[-1] == 3:
            y_field = y_field[:, :, :, np.newaxis, :]

        img = nib.Nifti1Image(y_field.astype(np.float32), t1w_affine, t1w_header)
        img.header.set_intent(1007)  # NIFTI_INTENT_VECTOR
        img.set_sform(t1w_affine, code=1)
        img.set_qform(t1w_affine, code=1)
        nib.save(img, str(p))

        if verbose:
            print(f"    y_{self.stem}.nii  [Forward deformation field]  "
                  f"{y_field.shape}")
        return str(p)

    def save_inverse_field(self,
                            iy_field:   np.ndarray,
                            mni_affine: np.ndarray,
                            t1w_header,
                            verbose:    bool = True) -> str:
        """
        Save iy_<stem>.nii — inverse deformation field.

        Inverse field lives on the MNI grid (91×109×91).
        """
        p = self._path("iy_")

        if iy_field.ndim == 4 and iy_field.shape[-1] == 3:
            iy_field = iy_field[:, :, :, np.newaxis, :]

        img = nib.Nifti1Image(iy_field.astype(np.float32), mni_affine, t1w_header)
        img.header.set_intent(1007)
        img.set_sform(mni_affine, code=1)
        img.set_qform(mni_affine, code=1)
        nib.save(img, str(p))

        if verbose:
            print(f"    iy_{self.stem}.nii  [Inverse deformation field]  "
                  f"{iy_field.shape}")
        return str(p)

    # ── combined save ─────────────────────────────────────────────────

    def save_all(self,
                 posteriors: np.ndarray,
                 corrected:  np.ndarray,
                 y_field:    np.ndarray,
                 iy_field:   np.ndarray,
                 t1w_affine: np.ndarray,
                 t1w_header,
                 mni_affine: np.ndarray,
                 verbose:    bool = True) -> dict:
        """
        Save all segmentation outputs. Returns dict of output paths.
        """
        if verbose:
            print("  Saving outputs...")

        tissue_paths = self.save_tissue_maps(
            posteriors, t1w_affine, t1w_header, verbose)
        m_path  = self.save_bias_corrected(
            corrected, t1w_affine, t1w_header, verbose)
        y_path  = self.save_forward_field(
            y_field, t1w_affine, t1w_header, verbose)
        iy_path = self.save_inverse_field(
            iy_field, mni_affine, t1w_header, verbose)

        return {
            "c1"  : tissue_paths[0],   # GM
            "c2"  : tissue_paths[1],   # WM
            "c3"  : tissue_paths[2],   # CSF
            "m"   : m_path,            # bias-corrected T1w
            "y"   : y_path,            # forward deformation field
            "iy"  : iy_path,           # inverse deformation field
        }