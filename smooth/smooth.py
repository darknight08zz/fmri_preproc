# -*- coding: utf-8 -*-
"""
smooth.py
══════════════════════════════════════════════════════════════════
SPM Smooth — Pipeline Orchestrator

Matches SPM batch exactly:
  FWHM            : [6 6 6] mm
  Data Type       : SAME
  Implicit masking: No
  Prefix          : s

Expected Output
───────────────
  swrarfunc_4D.nii  — smoothed functional in MNI 2mm space

Usage
─────
  python smooth.py --func wrarfunc_4D.nii
  python smooth.py --func wrarfunc_4D.nii --fwhm 6 6 6

  from smooth import run_smooth
  out = run_smooth(func_path)
══════════════════════════════════════════════════════════════════
"""

import sys, io, argparse, time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

from smooth_apply import Smoother


def run_smooth(
    func_path: str,
    fwhm_mm:   list = None,
    verbose:   bool = True,
) -> dict:
    """
    Run SPM Smooth on a 4D NIfTI.

    Parameters
    ----------
    func_path : path to wrarfunc_4D.nii
    fwhm_mm   : [fx, fy, fz] FWHM in mm. Default [6, 6, 6].

    Returns
    -------
    dict:
      output_file : str   path to swrarfunc_4D.nii
      time_total  : float seconds
    """
    if fwhm_mm is None:
        fwhm_mm = [6.0, 6.0, 6.0]

    t_start = time.time()

    if verbose:
        print("\n" + "=" * 60)
        print("  SPM SMOOTH")
        print("=" * 60)
        print(f"  Input  : {Path(func_path).name}")
        print(f"  FWHM   : {fwhm_mm} mm")
        print(f"  Data   : SAME (float32)")
        print(f"  Mask   : No implicit masking")
        print(f"  Prefix : s")
        print()

    smoother = Smoother(fwhm_mm=fwhm_mm)
    out_path = smoother.run(func_path=func_path, verbose=verbose)

    t_total = time.time() - t_start

    if verbose:
        print("\n" + "=" * 60)
        print("  SMOOTHING COMPLETE")
        print("=" * 60)
        print(f"  Output : {Path(out_path).name}")
        print(f"  Total  : {t_total:.1f}s")
        print("=" * 60 + "\n")

    print(f"__RESULT__ {out_path}")

    return {
        "output_file" : out_path,
        "time_total"  : round(t_total, 2),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="SPM Smooth")
    p.add_argument("--func", required=True,
                   help="Normalised functional image (wrarfunc_4D.nii)")
    p.add_argument("--fwhm", nargs=3, type=float, default=[6.0, 6.0, 6.0],
                   metavar=("FX", "FY", "FZ"),
                   help="FWHM in mm per axis (default: 6 6 6)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_smooth(
        func_path = args.func,
        fwhm_mm   = args.fwhm,
        verbose   = not args.quiet,
    )