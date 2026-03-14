# -*- coding: utf-8 -*-
"""
normalise_apply.py
══════════════════════════════════════════════════════════════════
SPM Normalise: Write — apply deformation field to fMRI volumes

Matches SPM batch exactly:
  Deformation Field : y_T1w.nii         (forward field, subject→MNI)
  Images to Write   : rarfunc_4D.nii
  Bounding Box      : [-78 78; -112 76; -70 85]  (MNI 2mm standard)
  Voxel Size        : [2 2 2] mm
  Interpolation     : 4th-degree B-spline
  Prefix            : w

Algorithm (SPM spm_write_sn.m):
  1. Build MNI 2mm output grid from bounding box
  2. Invert y_ field: for each MNI output voxel find subject-space mm
  3. Convert subject mm → fractional func voxel coords
  4. B-spline interpolate each volume at those coords
  5. Save wrarfunc_4D.nii with MNI 2mm affine
══════════════════════════════════════════════════════════════════
"""

import sys, io
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# ── MNI 2mm standard space constants ─────────────────────────────────
# Bounding box matches SPM default: [-78 78; -112 76; -70 85]
MNI_BB = np.array([[-78, 78],
                   [-112, 76],
                   [-70,  85]], dtype=np.float64)   # mm, (3,2)

MNI_VOX_SIZE = np.array([2.0, 2.0, 2.0])           # mm

# Standard MNI 2mm affine (matches spm_get_space for MNI152)
# Origin at voxel (40,57,46) in 1-based SPM convention → (39,56,45) 0-based
MNI_AFFINE = np.array([
    [-2.,  0.,  0.,  78.],
    [ 0.,  2.,  0., -112.],
    [ 0.,  0.,  2., -70.],
    [ 0.,  0.,  0.,   1.],
], dtype=np.float64)


def _mni_output_shape() -> tuple:
    """
    Compute output grid shape from bounding box and voxel size.
    shape[d] = floor((BB[d,1] - BB[d,0]) / vox + 1)
    Standard MNI 2mm → (79, 95, 79)
    """
    shape = tuple(
        int(np.round((MNI_BB[d, 1] - MNI_BB[d, 0]) / MNI_VOX_SIZE[d])) + 1
        for d in range(3)
    )
    return shape


def _build_mni_world_coords(out_shape: tuple) -> np.ndarray:
    """
    Build (3, N) array of MNI world coordinates (mm) for every output voxel.
    MNI voxel (i,j,k) → world = MNI_AFFINE @ [i,j,k,1]
    """
    i = np.arange(out_shape[0])
    j = np.arange(out_shape[1])
    k = np.arange(out_shape[2])
    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")
    vox_hom = np.vstack([ii.ravel(), jj.ravel(), kk.ravel(),
                          np.ones(ii.size)])                    # (4, N)
    world = MNI_AFFINE @ vox_hom                                # (4, N)
    return world[:3]                                            # (3, N)


def _invert_deformation_field(y_field:    np.ndarray,
                               y_affine:   np.ndarray,
                               mni_coords: np.ndarray,
                               verbose:    bool = True) -> np.ndarray:
    """
    Invert the forward deformation field y_ to map MNI mm → subject mm.

    y_field shape: (X,Y,Z,1,3) or (X,Y,Z,3)
      Each voxel stores the MNI world coordinates (mm) that subject
      voxel maps to.

    Strategy (SPM spm_def2det / scattered interpolation):
      - Treat the 3 component volumes of y_ as 3 separate scalar fields
        defined on the subject voxel grid
      - For each MNI output coordinate, interpolate the INVERSE mapping
        by using RegularGridInterpolator on each component
      - This gives, for each MNI voxel, the subject-space mm coordinates

    Parameters
    ----------
    y_field    : (X,Y,Z,1,3) or (X,Y,Z,3) deformation field
    y_affine   : (4,4) affine of the y_ NIfTI (= T1w affine)
    mni_coords : (3,N) MNI world coordinates to map back

    Returns
    -------
    subj_mm : (3,N) subject-space mm coordinates
    """
    # Squeeze to (X,Y,Z,3)
    if y_field.ndim == 5:
        y_field = y_field[:, :, :, 0, :]   # (X,Y,Z,3)

    shape = y_field.shape[:3]
    if verbose:
        print(f"    y_ field shape : {y_field.shape}")

    # Subject voxel grid axes (0-based)
    xi = np.arange(shape[0], dtype=np.float64)
    yi = np.arange(shape[1], dtype=np.float64)
    zi = np.arange(shape[2], dtype=np.float64)

    # The y_ field stores MNI mm at each subject voxel.
    # We need the reverse: given MNI mm, find subject voxel.
    #
    # Approach: build 3 interpolators — one per MNI component stored in y_.
    # y_field[...,0] = MNI_x mm at each subject voxel
    # y_field[...,1] = MNI_y mm at each subject voxel
    # y_field[...,2] = MNI_z mm at each subject voxel
    #
    # Then build inverse maps using the subject voxel-to-world affine:
    # For each subject voxel (i,j,k), compute its world mm via y_affine,
    # and invert by finding which subject voxel corresponds to each MNI coord.
    #
    # Practical inversion: use the subject voxel grid world coords as
    # the "values" and y_ components as the "points" — i.e. treat
    # y_[i,j,k] = MNI_mm as the forward mapping, and interpolate
    # subject_world_mm as a function of MNI_mm using scattered interp.
    #
    # For efficiency we use component-wise trilinear interpolation on the
    # subject grid, evaluating at MNI coords re-expressed as subject voxels.

    # Step 1: compute subject world mm for every subject voxel
    ii, jj, kk = np.meshgrid(xi, yi, zi, indexing="ij")
    subj_vox_hom = np.vstack([ii.ravel(), jj.ravel(), kk.ravel(),
                               np.ones(ii.size)])               # (4,N)
    subj_world   = (y_affine @ subj_vox_hom)[:3]               # (3,N) mm

    # Step 2: for each MNI output coord, find nearest subject voxel
    # by inverting the y_ field components via nearest-neighbour on the
    # MNI mm grid stored in y_, then refining with trilinear interpolation.
    #
    # Build interpolators: subject_world_mm[d] as function of subject voxel
    interp_sx = RegularGridInterpolator(
        (xi, yi, zi), subj_world[0].reshape(shape),
        method="linear", bounds_error=False, fill_value=None)
    interp_sy = RegularGridInterpolator(
        (xi, yi, zi), subj_world[1].reshape(shape),
        method="linear", bounds_error=False, fill_value=None)
    interp_sz = RegularGridInterpolator(
        (xi, yi, zi), subj_world[2].reshape(shape),
        method="linear", bounds_error=False, fill_value=None)

    # Step 3: invert y_ field
    # y_field stores, at subject voxel (i,j,k), the MNI mm of that voxel.
    # Build interpolators for each component of subject_mm as a function
    # of the MNI mm stored in y_:
    # We want: given mni_coord → what subject_mm?
    # Equivalent to: interp subj_world over the MNI mm grid from y_
    #
    # The y_ field itself IS the forward mapping sampled on subject grid.
    # Inversion: for each MNI query point, find the subject voxel (i,j,k)
    # whose y_[i,j,k] is closest to the query MNI point.
    # Use iterative fixed-point or direct scattered interpolation.
    #
    # SPM uses an iterative approach. We use scipy's scattered interpolation
    # on a subsampled grid for speed, then refine.

    # Build scattered interpolators: MNI_mm → subj_mm
    # Subsample to keep memory reasonable
    n_subj = int(np.prod(shape))
    step   = max(1, n_subj // 500_000)   # subsample if large

    mni_pts  = np.stack([
        y_field[:, :, :, 0].ravel()[::step],
        y_field[:, :, :, 1].ravel()[::step],
        y_field[:, :, :, 2].ravel()[::step],
    ], axis=1)                                                   # (M,3)

    subj_pts = np.stack([
        subj_world[0][::step],
        subj_world[1][::step],
        subj_world[2][::step],
    ], axis=1)                                                   # (M,3)

    from scipy.interpolate import LinearNDInterpolator
    if verbose:
        print(f"    Building inverse field interpolator "
              f"({mni_pts.shape[0]:,} control points)...")

    interp_inv = LinearNDInterpolator(mni_pts, subj_pts, fill_value=np.nan)

    # Evaluate at MNI query coords
    query_pts  = mni_coords.T                                    # (N,3)
    subj_mm    = interp_inv(query_pts).T                         # (3,N)

    # Fill NaN (outside convex hull) with nearest neighbour
    nan_mask = np.isnan(subj_mm[0])
    if nan_mask.any():
        from scipy.spatial import cKDTree
        tree      = cKDTree(mni_pts)
        _, idx    = tree.query(query_pts[nan_mask])
        subj_mm[:, nan_mask] = subj_pts[idx].T
        if verbose:
            print(f"    NaN fill (outside FOV): {nan_mask.sum():,} voxels")

    return subj_mm                                               # (3,N)


def _bspline_interpolate_volume(vol:        np.ndarray,
                                 subj_vox:   np.ndarray) -> np.ndarray:
    """
    Sample a 3D volume at fractional voxel coordinates using
    4th-degree B-spline interpolation (matches SPM interpolation=4).

    Parameters
    ----------
    vol      : (X,Y,Z) float32 volume
    subj_vox : (3,N) fractional voxel coordinates

    Returns
    -------
    sampled : (N,) interpolated values
    """
    # map_coordinates with order=4 = 4th-degree B-spline
    sampled = map_coordinates(
        vol.astype(np.float64),
        subj_vox,
        order=4,
        mode="constant",
        cval=0.0,
        prefilter=True    # prefilter=True required for B-spline order>1
    )
    return sampled.astype(np.float32)


class NormaliseWriter:
    """
    SPM Normalise: Write — warps images to MNI 2mm space.
    """

    def __init__(self,
                 y_path:   str,
                 func_path: str):
        self.y_path    = Path(y_path)
        self.func_path = Path(func_path)

        if not self.y_path.exists():
            raise FileNotFoundError(f"Deformation field not found: {self.y_path}")
        if not self.func_path.exists():
            raise FileNotFoundError(f"Functional image not found: {self.func_path}")

    def run(self, verbose: bool = True) -> str:
        """
        Apply deformation field and write wrarfunc_4D.nii.

        Returns path to output file.
        """
        import time
        t0 = time.time()

        # ── Load y_ deformation field ─────────────────────────────────
        if verbose:
            print("  Loading deformation field...")
        y_img    = nib.load(str(self.y_path))
        y_field  = y_img.get_fdata(dtype="float32")
        y_affine = y_img.affine.copy()

        if verbose:
            print(f"    y_ shape  : {y_field.shape}")
            print(f"    y_ affine : vox size = "
                  f"{np.sqrt((y_affine[:3,:3]**2).sum(0)).round(3)} mm")

        # ── Load functional 4D ────────────────────────────────────────
        if verbose:
            print("  Loading functional image...")
        func_img    = nib.load(str(self.func_path))
        func_data   = func_img.get_fdata(dtype="float32")
        func_affine = func_img.affine.copy()
        func_header = func_img.header

        if func_data.ndim == 3:
            func_data = func_data[..., np.newaxis]
        n_vols = func_data.shape[3]

        if verbose:
            print(f"    Func shape : {func_data.shape}  ({n_vols} volumes)")

        # ── Build MNI output grid ─────────────────────────────────────
        out_shape  = _mni_output_shape()
        n_out      = int(np.prod(out_shape))

        if verbose:
            print(f"  MNI output grid : {out_shape}  "
                  f"({n_out:,} voxels per volume)")

        mni_coords = _build_mni_world_coords(out_shape)         # (3,N) mm

        # ── Invert deformation field ──────────────────────────────────
        if verbose:
            print("  Inverting deformation field (MNI → subject mm)...")
        subj_mm = _invert_deformation_field(
            y_field, y_affine, mni_coords, verbose=verbose
        )

        # ── Convert subject mm → fractional func voxel coords ────────
        if verbose:
            print("  Converting subject mm → functional voxel coords...")
        func_aff_inv = np.linalg.inv(func_affine)
        subj_mm_hom  = np.vstack([subj_mm, np.ones((1, n_out))])   # (4,N)
        func_vox     = (func_aff_inv @ subj_mm_hom)[:3]            # (3,N)

        # ── Warp each volume ──────────────────────────────────────────
        if verbose:
            print(f"  Warping {n_vols} volumes with 4th-degree B-spline...")

        out_data = np.zeros((*out_shape, n_vols), dtype=np.float32)

        for v in range(n_vols):
            if verbose and (v % 50 == 0 or v == n_vols - 1):
                print(f"    Volume {v+1:4d}/{n_vols}")
            out_data[..., v] = _bspline_interpolate_volume(
                func_data[..., v], func_vox
            ).reshape(out_shape)

        # ── Save output ───────────────────────────────────────────────
        out_path = self.func_path.parent / ("w" + self.func_path.name)

        out_img = nib.Nifti1Image(out_data, MNI_AFFINE)
        out_img.set_sform(MNI_AFFINE, code=1)
        out_img.set_qform(MNI_AFFINE, code=1)

        # Preserve TR from original header
        try:
            tr = float(func_header.get_zooms()[3])
            if tr > 0:
                out_img.header.set_zooms((*MNI_VOX_SIZE, tr))
        except Exception:
            pass

        nib.save(out_img, str(out_path))

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  Output : {out_path.name}")
            print(f"  Shape  : {out_data.shape}")
            print(f"  Affine :\n{MNI_AFFINE}")
            print(f"  Time   : {elapsed:.1f}s")

        return str(out_path)