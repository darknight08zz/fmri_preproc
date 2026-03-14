# -*- coding: utf-8 -*-
"""
coregister_reslice.py
══════════════════════════════════════════════════════════════════
SPM-matched Coregistration — Reslice step

Matches SPM's spm_reslice.m behaviour:
  - Interpolation : 4th-degree B-spline  (your SPM batch setting)
  - Wrapping      : No wrap  [0, 0, 0]
  - Masking       : Don't mask (your SPM batch setting)
  - Prefix        : "r"

Inputs (matching your SPM batch):
  source           = meansub-02_4D.nii   (has transform in header)
  other_images     = arsub-02_4D.nii     (receives same transform)
  reference        = T1w.nii             (defines output voxel grid)

Outputs:
  rmeansub-02_4D.nii   ← resliced mean functional
  rarsub-02_4D.nii     ← resliced realigned 4D functional
══════════════════════════════════════════════════════════════════
"""

import sys, io
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
from pathlib import Path
from typing import List, Union

# ── Force UTF-8 stdout on Windows ────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")


# ──────────────────────────────────────────────────────────────
#  Constants (mirrors your SPM batch)
# ──────────────────────────────────────────────────────────────

INTERP_ORDER = 4          # 4th-degree B-spline (SPM default for EPI)
WRAP         = [0, 0, 0]  # No wrapping
MASK         = False       # Don't mask (your setting)
PREFIX       = "r"


# ──────────────────────────────────────────────────────────────
#  Core reslice function
# ──────────────────────────────────────────────────────────────

def _reslice_volume(vol_data:    np.ndarray,
                    vol_affine:  np.ndarray,
                    ref_affine:  np.ndarray,
                    ref_shape:   tuple,
                    interp_order: int  = INTERP_ORDER,
                    wrap:         list = WRAP,
                    mask:         bool = MASK) -> np.ndarray:
    """
    Reslice a single 3D volume onto the reference voxel grid.

    Strategy (identical to spm_reslice.m):
    ─────────────────────────────────────
    For each output voxel o in reference space:
      1. Convert o → world mm via ref_affine
      2. Convert world mm → source voxel via inv(vol_affine)
         Note: vol_affine already has the coreg transform baked in
               (written by write_to_header in coregister_estimate.py)
      3. Sample source at those voxel coords with B-spline interp

    Parameters
    ----------
    vol_data    : (X, Y, Z) source volume
    vol_affine  : (4,4) source affine WITH coreg transform baked in
    ref_affine  : (4,4) reference (T1w) affine — defines output grid
    ref_shape   : (X, Y, Z) output grid shape
    interp_order: B-spline order (4 = SPM default)
    wrap        : [wx, wy, wz] wrap flags
    mask        : if True, zero out voxels outside source FOV

    Returns
    -------
    (X, Y, Z) resliced volume on the reference grid
    """

    # ── Build output voxel grid in reference space ────────────────────
    rx, ry, rz = ref_shape
    xi, yi, zi = np.meshgrid(np.arange(rx),
                              np.arange(ry),
                              np.arange(rz), indexing='ij')

    ref_vox_hom = np.vstack([xi.ravel(), yi.ravel(), zi.ravel(),
                              np.ones(xi.size)])         # (4, N)

    # ── ref voxel → world (mm) → source voxel ────────────────────────
    # world_mm = ref_affine @ ref_vox
    # src_vox  = inv(vol_affine) @ world_mm
    #
    # vol_affine already has M_coreg baked in:
    #   vol_affine = M_coreg @ original_vol_affine
    # So inv(vol_affine) automatically undoes the coreg transform.
    vol_affine_inv = np.linalg.inv(vol_affine)
    M_ref_to_src   = vol_affine_inv @ ref_affine           # (4,4)

    src_vox = M_ref_to_src @ ref_vox_hom                  # (4, N)
    src_vox = src_vox[:3]                                  # (3, N)

    # ── Handle wrapping (all False for your config) ───────────────────
    for ax, w in enumerate(wrap):
        if w:
            src_vox[ax] %= vol_data.shape[ax]

    # ── B-spline interpolation (order=4, matching SPM) ────────────────
    # prefilter=True is required for B-spline orders > 1
    # scipy applies the B-spline prefilter automatically when order > 1
    resliced = map_coordinates(
        vol_data,
        src_vox,
        order      = interp_order,
        mode       = 'constant',   # outside FOV → 0 (no wrap)
        cval       = 0.0,
        prefilter  = (interp_order > 1),   # needed for B-spline
    )

    # ── Masking (disabled in your SPM batch) ──────────────────────────
    if mask:
        # Zero out voxels where source coordinates were outside FOV
        sh = vol_data.shape
        outside = (
            (src_vox[0] < 0) | (src_vox[0] > sh[0] - 1) |
            (src_vox[1] < 0) | (src_vox[1] > sh[1] - 1) |
            (src_vox[2] < 0) | (src_vox[2] > sh[2] - 1)
        )
        resliced[outside] = 0.0

    return resliced.reshape(ref_shape).astype(np.float32)


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

class CoregReslicer:
    """
    SPM-matched Coregistration — Reslice step.

    Reslices source + other_images onto the reference grid using
    the transform already baked into each image's NIfTI header
    by CoregEstimator.write_to_header().

    Parameters match your SPM batch:
      interpolation = 4  (4th-degree B-spline)
      wrapping      = [0, 0, 0]  (No wrap)
      masking       = False  (Don't mask)
      prefix        = "r"
    """

    def __init__(self,
                 interp_order: int  = INTERP_ORDER,
                 wrap:         list = None,
                 mask:         bool = MASK,
                 prefix:       str  = PREFIX):

        self.interp_order = interp_order
        self.wrap         = wrap if wrap is not None else WRAP
        self.mask         = mask
        self.prefix       = prefix

    def reslice(self,
                ref_nii_path:    str,
                images_to_reslice: List[str],
                verbose:         bool = True) -> List[str]:
        """
        Reslice a list of images onto the reference grid.

        Parameters
        ----------
        ref_nii_path       : T1w.nii  — defines output voxel grid + affine
        images_to_reslice  : list of paths whose headers contain the
                             coreg transform (written by write_to_header).
                             Your batch:
                               [meansub-02_4D.nii, arsub-02_4D.nii]
        verbose            : print progress

        Returns
        -------
        List of output file paths (with prefix "r" prepended)
        """
        if verbose:
            print("═" * 60)
            print("  COREGISTRATION — RESLICE")
            print("═" * 60)
            print(f"  Reference   : {Path(ref_nii_path).name}")
            print(f"  Interpolation: {self.interp_order}th-degree B-spline")
            print(f"  Wrapping    : {self.wrap}")
            print(f"  Masking     : {self.mask}")
            print(f"  Prefix      : '{self.prefix}'")

        # Load reference — defines the output voxel grid
        ref_img    = nib.load(ref_nii_path)
        ref_affine = ref_img.affine.copy()
        ref_shape  = ref_img.shape[:3]

        if verbose:
            print(f"\n  Reference grid : {ref_shape}  affine det={np.linalg.det(ref_affine[:3,:3]):.2f}")

        output_paths = []

        for img_path in images_to_reslice:
            img_path = str(img_path)
            p        = Path(img_path)

            if verbose:
                print(f"\n  Reslicing : {p.name}")

            src_img    = nib.load(img_path)
            src_affine = src_img.affine.copy()   # has coreg M baked in
            src_data   = np.asarray(src_img.get_fdata(dtype='float32'))
            src_header = src_img.header

            is_4d = src_data.ndim == 4

            if is_4d:
                n_vols = src_data.shape[3]
                if verbose:
                    print(f"    4D volume — reslicing {n_vols} frames ...")

                resliced_4d = np.zeros((*ref_shape, n_vols), dtype=np.float32)

                for t in range(n_vols):
                    if verbose and t % 20 == 0:
                        print(f"    Volume {t+1}/{n_vols}", end='\r')

                    resliced_4d[..., t] = _reslice_volume(
                        vol_data     = src_data[..., t],
                        vol_affine   = src_affine,
                        ref_affine   = ref_affine,
                        ref_shape    = ref_shape,
                        interp_order = self.interp_order,
                        wrap         = self.wrap,
                        mask         = self.mask,
                    )

                if verbose:
                    print(f"    Volume {n_vols}/{n_vols} ✓")

                out_data = resliced_4d

            else:
                # 3D (mean functional)
                out_data = _reslice_volume(
                    vol_data     = src_data,
                    vol_affine   = src_affine,
                    ref_affine   = ref_affine,
                    ref_shape    = ref_shape,
                    interp_order = self.interp_order,
                    wrap         = self.wrap,
                    mask         = self.mask,
                )

            # ── Save output with prefix ───────────────────────────────
            out_name = self.prefix + p.name
            out_path = p.parent / out_name

            # Output uses reference affine (image is now in T1w space)
            out_img = nib.Nifti1Image(out_data, ref_affine, src_header)
            out_img.set_sform(ref_affine, code=1)
            out_img.set_qform(ref_affine, code=1)
            nib.save(out_img, str(out_path))

            output_paths.append(str(out_path))

            if verbose:
                print(f"    ✓ Saved → {out_path.name}")

        if verbose:
            print("\n  ✓ Reslice complete")
            print("  Output files:")
            for op in output_paths:
                print(f"    {Path(op).name}")

        return output_paths


# ──────────────────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python coregister_reslice.py <T1w.nii> <meansub-02_4D.nii> <arsub-02_4D.nii>")
        sys.exit(1)

    reslicer = CoregReslicer(
        interp_order = 4,
        wrap         = [0, 0, 0],
        mask         = False,
        prefix       = "r",
    )

    out = reslicer.reslice(
        ref_nii_path      = sys.argv[1],
        images_to_reslice = sys.argv[2:],
    )
    print(f"\nDone. Outputs: {out}")