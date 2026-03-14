# -*- coding: utf-8 -*-
"""
coregister_estimate.py
══════════════════════════════════════════════════════════════════
SPM-matched Coregistration — Estimate step

Matches SPM's spm_coreg.m behaviour:
  - Cost function : Normalised Mutual Information (NMI)
  - Optimiser     : Powell (scipy) ≈ SPM's spm_powell
  - Multi-res     : separation = [4, 2] mm (coarse → fine)
  - Joint hist    : 256 bins, FWHM=[7,7] smoothing
  - Tolerances    : [0.02 0.02 0.02  0.001 0.001 0.001]
  - Rotation order: Rx @ Ry @ Rz  (SPM XYZ convention)
  - World-space   : all operations in mm via affine

Inputs (matching your SPM batch):
  reference  = T1w.nii            (fixed — stays still)
  source     = meansub-02_4D.nii  (moving — mean functional)

Output:
  6-parameter rigid vector [tx,ty,tz,rx,ry,rz]
  + 4×4 world-space affine M_src_to_ref
  Written into source NIfTI header (sform/qform) — same as SPM
══════════════════════════════════════════════════════════════════
"""

import sys, io
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from pathlib import Path

# ── Force UTF-8 stdout on Windows ────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")


# ──────────────────────────────────────────────────────────────
#  Constants (mirrors SPM defaults)
# ──────────────────────────────────────────────────────────────

SEPARATION_MM   = [4.0, 2.0]          # multi-resolution passes
HIST_BINS       = 256                  # joint histogram bins
HIST_SMOOTH_FWHM = [7.0, 7.0]         # FWHM for histogram smoothing (mm → sigma below)
TOLERANCES      = [0.02, 0.02, 0.02,  # tx, ty, tz  (mm)
                   0.001, 0.001, 0.001]  # rx, ry, rz (radians)
INTERP_ORDER    = 1                    # trilinear during estimation (speed)


# ──────────────────────────────────────────────────────────────
#  Rotation matrix (SPM XYZ: Rx @ Ry @ Rz)
# ──────────────────────────────────────────────────────────────

def _build_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """3×3 rotation matrix in SPM XYZ convention."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1,  0,   0 ],
                   [0,  cx, -sx],
                   [0,  sx,  cx]])

    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])

    return Rx @ Ry @ Rz   # SPM order


def _params_to_M(params: np.ndarray) -> np.ndarray:
    """
    Convert 6-parameter rigid vector → 4×4 world-space affine.
    params = [tx, ty, tz, rx, ry, rz]
    M maps SOURCE world coords → REFERENCE world coords.
    """
    tx, ty, tz, rx, ry, rz = params
    R = _build_R(rx, ry, rz)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3,  3] = [tx, ty, tz]
    return M


# ──────────────────────────────────────────────────────────────
#  Joint histogram + NMI
# ──────────────────────────────────────────────────────────────

def _smooth_hist(h: np.ndarray, fwhm: list) -> np.ndarray:
    """
    Smooth 2-D joint histogram with a Gaussian kernel.
    fwhm = [fwhm_ref, fwhm_src] in histogram-bin units.
    """
    sigma = [f / (2.0 * np.sqrt(2.0 * np.log(2.0))) for f in fwhm]
    return gaussian_filter(h.astype(np.float64), sigma=sigma)


def _joint_histogram(ref_vals: np.ndarray,
                     src_vals: np.ndarray,
                     n_bins: int = HIST_BINS) -> np.ndarray:
    """
    Build a joint histogram of (ref_vals, src_vals).
    Both arrays are normalised to [0, n_bins-1] before binning.
    """
    def _normalise(v):
        lo, hi = np.percentile(v, 0.1), np.percentile(v, 99.9)
        v = np.clip(v, lo, hi)
        return (v - lo) / (hi - lo + 1e-12) * (n_bins - 1)

    r = _normalise(ref_vals).astype(np.int32)
    s = _normalise(src_vals).astype(np.int32)

    h = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(h, (r, s), 1)
    return h


def _nmi(h: np.ndarray) -> float:
    """
    Normalised Mutual Information from smoothed joint histogram.
    NMI = (H(ref) + H(src)) / H(ref, src)   [Studholme 1999]
    We NEGATE it because scipy.minimize minimises.
    """
    h  = h / (h.sum() + 1e-12)
    h  = np.maximum(h, 1e-12)           # avoid log(0)

    hr = h.sum(axis=1)                  # marginal ref
    hs = h.sum(axis=0)                  # marginal src

    H_ref = -np.sum(hr * np.log(hr + 1e-12))
    H_src = -np.sum(hs * np.log(hs + 1e-12))
    H_jnt = -np.sum(h  * np.log(h  + 1e-12))

    nmi = (H_ref + H_src) / (H_jnt + 1e-12)
    return -nmi   # negate → minimise


# ──────────────────────────────────────────────────────────────
#  Sampling at a given resolution
# ──────────────────────────────────────────────────────────────

def _sample_at_separation(img_data: np.ndarray,
                           affine: np.ndarray,
                           sep_mm: float) -> tuple:
    """
    Return (world_coords, intensities) for voxels sampled
    every sep_mm millimetres in each direction.

    world_coords : (3, N) array in mm
    intensities  : (N,) array
    """
    vox_size = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))   # mm per voxel
    step     = np.maximum(np.round(sep_mm / vox_size).astype(int), 1)

    xs = np.arange(0, img_data.shape[0], step[0])
    ys = np.arange(0, img_data.shape[1], step[1])
    zs = np.arange(0, img_data.shape[2], step[2])

    xi, yi, zi = np.meshgrid(xs, ys, zs, indexing='ij')
    vox_coords  = np.vstack([xi.ravel(), yi.ravel(), zi.ravel(),
                              np.ones(xi.size)])            # (4, N)

    world_coords = (affine @ vox_coords)[:3]               # (3, N) in mm
    intensities  = img_data[xi.ravel(), yi.ravel(), zi.ravel()]

    # Keep only foreground voxels (brain mask — 5th percentile threshold)
    thresh = np.percentile(intensities[intensities > 0], 5) \
             if intensities.max() > 0 else 0
    mask   = intensities > thresh

    return world_coords[:, mask], intensities[mask]


def _sample_source_at_world(src_data: np.ndarray,
                             src_affine: np.ndarray,
                             world_coords: np.ndarray) -> np.ndarray:
    """
    Sample src_data at world_coords (mm) using trilinear interpolation.
    world_coords : (3, N)
    Returns      : (N,) intensity array
    """
    from scipy.ndimage import map_coordinates

    src_affine_inv = np.linalg.inv(src_affine)
    vox = (src_affine_inv @ np.vstack([world_coords,
                                        np.ones(world_coords.shape[1])]))[:3]

    vals = map_coordinates(src_data, vox, order=INTERP_ORDER,
                           mode='constant', cval=0, prefilter=False)
    return vals


# ──────────────────────────────────────────────────────────────
#  Cost function (NMI at one resolution pass)
# ──────────────────────────────────────────────────────────────

def _make_cost_fn(ref_world: np.ndarray,
                  ref_vals:  np.ndarray,
                  src_data:  np.ndarray,
                  src_affine_orig: np.ndarray,
                  hist_smooth_bins: list):
    """
    Returns a closure: params → scalar NMI cost.

    ref_world       : (3, N) reference sample positions in mm
    ref_vals        : (N,)   reference intensities at those positions
    src_data        : 3-D source volume array
    src_affine_orig : 4×4 source-image affine (before any transform)
    hist_smooth_bins: [sigma_ref, sigma_src] in bin units for joint hist
    """
    def cost(params):
        # M maps source world → reference world
        # To find where reference points land in *source* voxel space:
        #   vox_src = inv(src_affine_orig) @ inv(M) @ ref_world_hom
        M     = _params_to_M(params)
        M_inv = np.linalg.inv(M)

        # Transform reference world coords into (updated) source world
        ref_hom    = np.vstack([ref_world, np.ones(ref_world.shape[1])])
        src_world  = (M_inv @ ref_hom)[:3]        # (3, N)

        # Sample source at those world positions
        src_vals = _sample_source_at_world(src_data, src_affine_orig,
                                           src_world)

        # Only keep voxels where source was valid (inside FOV)
        valid = src_vals > 0
        if valid.sum() < 100:
            return 0.0    # degenerate — return neutral cost

        h    = _joint_histogram(ref_vals[valid], src_vals[valid])
        h_sm = _smooth_hist(h, hist_smooth_bins)
        return _nmi(h_sm)

    return cost


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

class CoregEstimator:
    """
    SPM-matched Coregistration — Estimate step.

    Parameters match your SPM batch exactly:
      separation       = [4, 2]              mm
      tolerances       = [0.02×3, 0.001×3]
      hist_smooth_fwhm = [7, 7]              mm (mapped to bin-sigma below)
      cost_fn          = NMI
    """

    def __init__(self,
                 separation:       list  = None,
                 tolerances:       list  = None,
                 hist_smooth_fwhm: list  = None):

        self.separation       = separation       or SEPARATION_MM
        self.tolerances       = tolerances       or TOLERANCES
        self.hist_smooth_fwhm = hist_smooth_fwhm or HIST_SMOOTH_FWHM

    # ── histogram smoothing: FWHM (mm) → bin sigma ───────────────────
    def _fwhm_to_binsigma(self, fwhm_mm: float, vox_range: float) -> float:
        """
        Convert FWHM in mm to sigma in histogram-bin units.
        vox_range = intensity range of the image (used to scale).
        SPM uses 7mm FWHM on the HISTOGRAM axis — here we approximate
        by converting to fraction of bin count.
        """
        fwhm_bins = (fwhm_mm / vox_range) * HIST_BINS
        return fwhm_bins / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def estimate(self,
                 ref_nii_path: str,
                 src_nii_path: str,
                 verbose:      bool = True) -> dict:
        """
        Estimate the rigid-body transform that aligns src → ref.

        Parameters
        ----------
        ref_nii_path : str   path to T1w.nii  (reference — fixed)
        src_nii_path : str   path to meansub-02_4D.nii  (source — moving)

        Returns
        -------
        dict with keys:
          params    : (6,) array [tx,ty,tz,rx,ry,rz]
          M         : (4,4) world-space affine src→ref
          ref_path  : str
          src_path  : str
        """
        if verbose:
            print("═" * 60)
            print("  COREGISTRATION — ESTIMATE")
            print("═" * 60)

        # ── Load images ───────────────────────────────────────────────
        ref_img = nib.load(ref_nii_path)
        src_img = nib.load(src_nii_path)

        ref_data   = np.asarray(ref_img.get_fdata(dtype='float32'))
        src_data   = np.asarray(src_img.get_fdata(dtype='float32'))
        ref_affine = ref_img.affine.copy()
        src_affine = src_img.affine.copy()

        # Handle 4D source (take mean — but mean*.nii should already be 3D)
        if src_data.ndim == 4:
            if verbose:
                print("  Source is 4D — taking mean volume for estimation")
            src_data = src_data.mean(axis=3)

        if verbose:
            print(f"  Reference : {Path(ref_nii_path).name}  {ref_data.shape}")
            print(f"  Source    : {Path(src_nii_path).name}  {src_data.shape}")

        # ── Histogram smoothing sigma (in bin units) ──────────────────
        ref_range = np.percentile(ref_data, 99.9) - np.percentile(ref_data, 0.1)
        src_range = np.percentile(src_data, 99.9) - np.percentile(src_data, 0.1)
        hist_smooth_bins = [
            self._fwhm_to_binsigma(self.hist_smooth_fwhm[0], ref_range),
            self._fwhm_to_binsigma(self.hist_smooth_fwhm[1], src_range),
        ]

        # ── Multi-resolution Powell optimisation ─────────────────────
        params = np.zeros(6, dtype=np.float64)   # start from identity

        for sep_mm in self.separation:
            if verbose:
                print(f"\n  Pass  sep={sep_mm:.0f} mm ──────────────────────────")

            # Sample reference at this resolution
            ref_world, ref_vals = _sample_at_separation(
                ref_data, ref_affine, sep_mm)

            if verbose:
                print(f"    Reference samples : {ref_world.shape[1]:,}")

            cost_fn = _make_cost_fn(ref_world, ref_vals,
                                    src_data, src_affine,
                                    hist_smooth_bins)

            # xtol/ftol ← tolerances (translation part)
            xtol = min(self.tolerances[:3])

            result = minimize(
                cost_fn,
                params,
                method  = 'Powell',
                options = {
                    'xtol'   : xtol,
                    'ftol'   : 1e-6,
                    'maxiter': 200,
                    'disp'   : False,
                },
            )
            params = result.x

            if verbose:
                t = params[:3]
                r = np.degrees(params[3:])
                print(f"    Translation  (mm) : x={t[0]:+.3f}  y={t[1]:+.3f}  z={t[2]:+.3f}")
                print(f"    Rotation    (deg) : x={r[0]:+.3f}  y={r[1]:+.3f}  z={r[2]:+.3f}")
                print(f"    NMI cost          : {result.fun:.6f}")

        # ── Final transform ───────────────────────────────────────────
        M = _params_to_M(params)

        if verbose:
            print("\n  ✓ Estimation complete")
            print(f"  Final params: {np.round(params, 4).tolist()}")

        return {
            'params'   : params,
            'M'        : M,
            'ref_path' : ref_nii_path,
            'src_path' : src_nii_path,
        }

    def write_to_header(self, src_nii_path: str, M: np.ndarray,
                        verbose: bool = True) -> None:
        """
        Write the estimated transform into the source image header.
        SPM stores coreg result in the sform/qform — same approach here.

        The new affine = M @ original_affine
        (i.e. source voxel → source world → reference world)
        """
        src_img    = nib.load(src_nii_path)
        # IMPORTANT windows fix for [Errno 22] Invalid argument:
        # We MUST completely detach from the file so we can overwrite it.
        # Even file_map and uncache isn't always enough due to memory mapping
        # under the hood. So we load the NIfTI from a memory buffer.
        import io
        with open(src_nii_path, 'rb') as f:
            file_bytes = f.read()
        
        # Load from memory to guarantee file handle is not held
        fh = nib.FileHolder(fileobj=io.BytesIO(file_bytes))
        src_img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        
        data       = np.asarray(src_img.dataobj)
        affine     = src_img.affine.copy()
        header     = src_img.header.copy()

        new_affine = M @ affine

        new_img = nib.Nifti1Image(
            data,
            new_affine,
            header,
        )
        new_img.set_sform(new_affine, code=1)
        new_img.set_qform(new_affine, code=1)
        nib.save(new_img, src_nii_path)

        if verbose:
            print(f"  ✓ Transform written to header: {Path(src_nii_path).name}")


# ──────────────────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python coregister_estimate.py <T1w.nii> <meansub-02_4D.nii>")
        sys.exit(1)

    estimator = CoregEstimator(
        separation       = [4.0, 2.0],
        tolerances       = [0.02, 0.02, 0.02, 0.001, 0.001, 0.001],
        hist_smooth_fwhm = [7.0, 7.0],
    )
    out = estimator.estimate(sys.argv[1], sys.argv[2])
    estimator.write_to_header(sys.argv[2], out['M'])
    print("\nDone.")