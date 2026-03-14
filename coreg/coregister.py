# -*- coding: utf-8 -*-
"""
coregister.py
══════════════════════════════════════════════════════════════════
SPM Coregistration: Estimate & Reslice — Full Pipeline Runner

Matches your SPM batch exactly:

  Image Selection
  ───────────────
  Reference  : T1w.nii
  Source     : meansub-02_4D.nii
  Other      : arsub-02_4D.nii

  Estimation Options
  ──────────────────
  Cost fn    : Normalised Mutual Information
  Separation : [4, 2] mm
  Tolerances : [0.02, 0.02, 0.02, 0.001, 0.001, 0.001]
  Hist FWHM  : [7, 7]

  Reslice Options
  ───────────────
  Interpolation : 4th-degree B-spline
  Wrapping      : No wrap  [0, 0, 0]
  Masking       : Don't mask
  Prefix        : "r"

  Expected outputs
  ────────────────
  rmeansub-02_4D.nii   ← resliced mean functional
  rarsub-02_4D.nii     ← resliced realigned 4D functional

Usage
─────
  python coregister.py \
      --ref    sub-02/anat/T1w.nii \
      --source sub-02/func/meansub-02_4D.nii \
      --other  sub-02/func/arsub-02_4D.nii

  # Or import and call from your pipeline:
  from coregister import run_coregistration
  outputs = run_coregistration(ref, source, [other])
══════════════════════════════════════════════════════════════════
"""

import argparse
import time
import sys
import io
from pathlib import Path

# ── FIX 1: Add this file's own directory to sys.path ────────────────
# When invoked via , __file__ is '<string>' not a real path.
# We guard against that — if __file__ is real, add its parent directory.
# If not (inline script mode), the route.ts wrapper already inserted the
# coreg/ directory via sys.path.insert before importing this module.
try:
    _THIS_DIR = str(Path(__file__).resolve().parent)
    if _THIS_DIR not in sys.path and '<' not in _THIS_DIR:
        sys.path.insert(0, _THIS_DIR)
except (NameError, TypeError):
    pass  # __file__ not available — route wrapper already set sys.path

# ── FIX 2: Force UTF-8 stdout on Windows ────────────────────────────
# Windows cmd.exe uses cp1252 which cannot encode box-drawing characters
# (= - checkmark degree) used in verbose output -> charmap codec error.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

from estimate import CoregEstimator
from reslice  import CoregReslicer


# ──────────────────────────────────────────────────────────────
#  Pipeline function
# ──────────────────────────────────────────────────────────────

def run_coregistration(
    ref_path:    str,
    source_path: str,
    other_paths: list  = None,
    # ── Estimation options (your SPM batch values) ──
    separation:        list  = None,
    tolerances:        list  = None,
    hist_smooth_fwhm:  list  = None,
    # ── Reslice options (your SPM batch values) ──
    interp_order:      int   = 4,
    wrap:              list  = None,
    mask:              bool  = False,
    prefix:            str   = "r",
    verbose:           bool  = True,
) -> dict:
    """
    Run full Coregistration: Estimate + Reslice.

    Parameters
    ----------
    ref_path     : T1w.nii  (reference — stays still)
    source_path  : meansub-02_4D.nii  (mean functional — moving)
    other_paths  : [arsub-02_4D.nii]  (gets same transform)

    Returns
    -------
    dict:
      params         : (6,) rigid params [tx,ty,tz,rx,ry,rz]
      M              : (4,4) transform matrix
      output_files   : list of resliced output paths
      time_estimate  : seconds
      time_reslice   : seconds
    """

    other_paths = other_paths or []

    if verbose:
        print("\n" + "═" * 60)
        print("  SPM COREGISTRATION: ESTIMATE & RESLICE")
        print("═" * 60)
        print(f"  Reference  : {Path(ref_path).name}")
        print(f"  Source     : {Path(source_path).name}")
        for o in other_paths:
            print(f"  Other      : {Path(o).name}")
        print()

    # ── STEP 1: ESTIMATE ─────────────────────────────────────────────
    t0 = time.time()

    estimator = CoregEstimator(
        separation       = separation       or [4.0, 2.0],
        tolerances       = tolerances       or [0.02, 0.02, 0.02, 0.001, 0.001, 0.001],
        hist_smooth_fwhm = hist_smooth_fwhm or [7.0, 7.0],
    )

    result = estimator.estimate(
        ref_nii_path = ref_path,
        src_nii_path = source_path,
        verbose      = verbose,
    )

    params = result['params']
    M      = result['M']

    # Write transform into source header (mean functional)
    estimator.write_to_header(source_path, M, verbose=verbose)

    # Write same transform into all Other Images headers (arsub-02_4D.nii)
    # SPM applies the SAME transform from the mean to all other images
    for other in other_paths:
        estimator.write_to_header(other, M, verbose=verbose)

    t_estimate = time.time() - t0

    if verbose:
        import numpy as np
        print(f"\n  Estimate time : {t_estimate:.1f} s")
        print(f"  tx={params[0]:+.3f} mm  ty={params[1]:+.3f} mm  tz={params[2]:+.3f} mm")
        print(f"  rx={np.degrees(params[3]):+.3f}°  ry={np.degrees(params[4]):+.3f}°  rz={np.degrees(params[5]):+.3f}°")

    # ── STEP 2: RESLICE ──────────────────────────────────────────────
    t1 = time.time()

    reslicer = CoregReslicer(
        interp_order = interp_order,
        wrap         = wrap or [0, 0, 0],
        mask         = mask,
        prefix       = prefix,
    )

    # Reslice source (mean) + all other images
    images_to_reslice = [source_path] + list(other_paths)

    output_files = reslicer.reslice(
        ref_nii_path       = ref_path,
        images_to_reslice  = images_to_reslice,
        verbose            = verbose,
    )

    t_reslice = time.time() - t1

    if verbose:
        print(f"\n  Reslice time : {t_reslice:.1f} s")
        print(f"  Total time   : {t_estimate + t_reslice:.1f} s")
        print("\n  ✓ COREGISTRATION COMPLETE")
        print("  Expected outputs:")
        for op in output_files:
            print(f"    {Path(op).name}")
        print("═" * 60 + "\n")

    return {
        'params'       : params,
        'M'            : M,
        'output_files' : output_files,
        'time_estimate': t_estimate,
        'time_reslice' : t_reslice,
    }


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="SPM-matched Coregistration: Estimate & Reslice"
    )
    p.add_argument("--ref",    required=True,
                   help="Reference image  (T1w.nii)")
    p.add_argument("--source", required=True,
                   help="Source image     (meansub-02_4D.nii)")
    p.add_argument("--other",  nargs="*", default=[],
                   help="Other images     (arsub-02_4D.nii ...)")

    # Estimation options
    p.add_argument("--sep",   nargs="+", type=float, default=[4.0, 2.0],
                   help="Separation mm    (default: 4 2)")
    p.add_argument("--fwhm",  nargs="+", type=float, default=[7.0, 7.0],
                   help="Hist smooth FWHM (default: 7 7)")

    # Reslice options
    p.add_argument("--interp", type=int, default=4,
                   help="B-spline order   (default: 4)")
    p.add_argument("--prefix", type=str, default="r",
                   help="Output prefix    (default: r)")
    p.add_argument("--mask",   action="store_true",
                   help="Apply masking    (default: False)")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    run_coregistration(
        ref_path    = args.ref,
        source_path = args.source,
        other_paths = args.other,
        separation  = args.sep,
        hist_smooth_fwhm = args.fwhm,
        interp_order = args.interp,
        prefix      = args.prefix,
        mask        = args.mask,
    )