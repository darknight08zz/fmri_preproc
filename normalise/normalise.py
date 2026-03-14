# -*- coding: utf-8 -*-
"""
normalise.py
══════════════════════════════════════════════════════════════════
SPM Normalise: Write — Pipeline Orchestrator

Matches SPM batch exactly:
  Deformation Field : y_T1w.nii
  Images to Write   : rarfunc_4D.nii
  Bounding Box      : [-78 78; -112 76; -70 85]
  Voxel Size        : [2 2 2] mm
  Interpolation     : 4th-degree B-spline
  Prefix            : w

Expected Output
───────────────
  wrarfunc_4D.nii  — functional image in MNI 2mm space

Usage
─────
  python normalise.py --y y_T1w.nii --func rarfunc_4D.nii

  from normalise import run_normalise
  out = run_normalise(y_path, func_path)
══════════════════════════════════════════════════════════════════
"""

import sys, io, argparse, time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

from normalise_apply import NormaliseWriter


def run_normalise(
    y_path:    str,
    func_path: str,
    verbose:   bool = True,
) -> dict:
    """
    Run SPM Normalise: Write.

    Parameters
    ----------
    y_path    : path to y_T1w.nii (forward deformation field)
    func_path : path to rarfunc_4D.nii

    Returns
    -------
    dict:
      output_file  : str   path to wrarfunc_4D.nii
      time_total   : float seconds
    """
    t_start = time.time()

    if verbose:
        print("\n" + "=" * 60)
        print("  SPM NORMALISE: WRITE")
        print("=" * 60)
        print(f"  Deformation field : {Path(y_path).name}")
        print(f"  Functional input  : {Path(func_path).name}")
        print(f"  Voxel size        : 2 x 2 x 2 mm")
        print(f"  Interpolation     : 4th-degree B-spline")
        print(f"  Prefix            : w")
        print()

    writer = NormaliseWriter(y_path=y_path, func_path=func_path)
    out_path = writer.run(verbose=verbose)

    t_total = time.time() - t_start

    if verbose:
        print("\n" + "=" * 60)
        print("  NORMALISATION COMPLETE")
        print("=" * 60)
        print(f"  Output : {Path(out_path).name}")
        print(f"  Total  : {t_total:.1f}s")
        print("=" * 60 + "\n")

    # Print result for route.ts to parse
    print(f"__RESULT__ {out_path}")

    return {
        "output_file" : out_path,
        "time_total"  : round(t_total, 2),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="SPM Normalise: Write")
    p.add_argument("--y",    required=True,
                   help="Forward deformation field  (y_T1w.nii)")
    p.add_argument("--func", required=True,
                   help="Functional image to warp   (rarfunc_4D.nii)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_normalise(
        y_path    = args.y,
        func_path = args.func,
        verbose   = not args.quiet,
    )