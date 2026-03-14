# -*- coding: utf-8 -*-
"""
segmentation.py
══════════════════════════════════════════════════════════════════
SPM Segmentation — Full Pipeline Orchestrator

Matches your SPM batch exactly:

  Channel
  ───────
  Volumes              : T1w.nii
  Bias Regularisation  : 0.0001
  Bias FWHM            : 60 mm
  Save Bias Corrected  : Save nothing (mT1w.nii saved separately)

  Tissue Classes
  ──────────────
  Tissue 1 GM    : TPM.nii,1   1 Gaussian   Native
  Tissue 2 WM    : TPM.nii,2   1 Gaussian   Native
  Tissue 3 CSF   : TPM.nii,3   2 Gaussians  Native
  Tissue 4 Bone  : TPM.nii,4   3 Gaussians  None
  Tissue 5 Soft  : TPM.nii,5   4 Gaussians  None
  Tissue 6 Air   : TPM.nii,6   2 Gaussians  None

  Warping & MRF
  ─────────────
  MRF Parameter        : 1
  Clean Up             : Light clean
  Warping Regularisation : [0 0.001 0.5 0.05 0.2]
  Affine Regularisation: European brains
  Smoothness           : 0
  Sampling Distance    : 3

  Deformation Fields   : Inverse + Forward

  Expected Outputs
  ────────────────
  c1T1w.nii   — Gray Matter
  c2T1w.nii   — White Matter
  c3T1w.nii   — CSF
  mT1w.nii    — Bias-corrected T1w
  y_T1w.nii   — Forward deformation field  (subject -> MNI)
  iy_T1w.nii  — Inverse deformation field  (MNI -> subject)

Usage
─────
  python segmentation.py --t1w T1w.nii --tpm <SPM_dir>/tpm/TPM.nii

  from segmentation import run_segmentation
  out = run_segmentation(t1w_path, tpm_path)
══════════════════════════════════════════════════════════════════
"""

import sys, io, argparse, time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

from segment_load   import SegmentLoader
from segment_bias   import BiasCorrector
from segment_gmm    import TissueGMM
from segment_mrf    import MRFCleanup
from segment_deform import DeformationEstimator
from segment_save   import SegmentSaver


# ──────────────────────────────────────────────────────────────────────
#  Pipeline function
# ──────────────────────────────────────────────────────────────────────

def run_segmentation(
    t1w_path:     str,
    tpm_path:     str,
    # ── Bias field (SPM batch values) ──
    bias_reg:     float = 0.0001,
    bias_fwhm:    float = 60.0,
    # ── GMM (SPM batch n_gauss values) ──
    n_gauss:      list  = None,
    # ── MRF (SPM batch values) ──
    mrf_strength: float = 1.0,
    # ── Deformation (SPM batch values) ──
    sampling_mm:  float = 3.0,
    # ── EM iterations ──
    n_em_iter:    int   = 30,
    bias_iter:    int   = 3,
    verbose:      bool  = True,
) -> dict:
    """
    Run full SPM-matched segmentation pipeline.

    Parameters
    ----------
    t1w_path     : path to T1w.nii
    tpm_path     : path to <SPM_dir>/tpm/TPM.nii

    Returns
    -------
    dict:
      output_files : dict  { c1, c2, c3, m, y, iy }  → file paths
      time_load    : float
      time_bias    : float
      time_gmm     : float
      time_mrf     : float
      time_deform  : float
      time_save    : float
      time_total   : float
      n_gm_voxels  : int
      n_wm_voxels  : int
      n_csf_voxels : int
    """
    import numpy as np

    if n_gauss is None:
        n_gauss = [1, 1, 2, 3, 4, 2]   # SPM defaults from your batch

    t_start = time.time()

    if verbose:
        print("\n" + "=" * 60)
        print("  SPM SEGMENTATION")
        print("=" * 60)
        print(f"  T1w : {Path(t1w_path).name}")
        print(f"  TPM : {Path(tpm_path).name}")
        print()

    # ── STEP 1: Load ──────────────────────────────────────────────────
    t0 = time.time()
    if verbose:
        print("[ Step 1 / 6 ]  Load Images")
    loader = SegmentLoader(t1w_path, tpm_path)
    ctx    = loader.load(verbose=verbose)
    t_load = time.time() - t0

    # ── STEP 2: Bias Correction (initial pass) ────────────────────────
    t0 = time.time()
    if verbose:
        print(f"\n[ Step 2 / 6 ]  Bias Field Estimation  "
              f"(reg={bias_reg}, FWHM={bias_fwhm}mm)")

    bias_corr = BiasCorrector(bias_reg=bias_reg, bias_fwhm=bias_fwhm)

    # First bias pass: use TPM as initial "posteriors" approximation
    init_posteriors = ctx["tpm_data"]   # shape (X,Y,Z,6)
    bias_field, corrected, _ = bias_corr.estimate(
        ctx["t1w_data"], init_posteriors, ctx["t1w_affine"],
        verbose=verbose, n_iter=bias_iter
    )
    t_bias = time.time() - t0

    # ── STEP 3: GMM tissue classification ─────────────────────────────
    t0 = time.time()
    if verbose:
        print(f"\n[ Step 3 / 6 ]  GMM Classification  "
              f"(n_gauss={n_gauss})")

    gmm = TissueGMM(
        n_gauss_per_class = n_gauss,
        n_iter            = n_em_iter,
        mrf_strength      = mrf_strength,
    )
    posteriors = gmm.fit(corrected, ctx["tpm_data"], verbose=verbose)
    t_gmm      = time.time() - t0

    # ── STEP 4: Refined bias correction using GMM posteriors ──────────
    if verbose:
        print(f"\n[ Step 4 / 6 ]  Refined Bias Correction  (using GMM posteriors)")
    _, corrected, _ = bias_corr.estimate(
        ctx["t1w_data"], posteriors, ctx["t1w_affine"],
        verbose=verbose, n_iter=bias_iter
    )

    # ── STEP 5: MRF cleanup ───────────────────────────────────────────
    t0 = time.time()
    if verbose:
        print(f"\n[ Step 5 / 6 ]  MRF Cleanup  "
              f"(strength={mrf_strength}, Light clean)")

    mrf        = MRFCleanup(mrf_strength=mrf_strength)
    posteriors = mrf.run(posteriors, verbose=verbose)
    t_mrf      = time.time() - t0

    # ── STEP 6a: Deformation fields ───────────────────────────────────
    t0 = time.time()
    if verbose:
        print(f"\n[ Step 6 / 6 ]  Deformation Fields  "
              f"(sampling={sampling_mm}mm, Fwd+Inv)")

    deformer  = DeformationEstimator(sampling_mm=sampling_mm)
    deform    = deformer.estimate(
        ctx["t1w_data"], ctx["t1w_affine"], ctx["tpm_data"], verbose=verbose
    )
    t_deform  = time.time() - t0

    # ── STEP 6b: Save all outputs ─────────────────────────────────────
    t0    = time.time()
    saver = SegmentSaver(t1w_path)
    if verbose:
        print(f"\n  Saving output files...")

    output_files = saver.save_all(
        posteriors  = posteriors,
        corrected   = corrected,
        y_field     = deform["y_field"],
        iy_field    = deform["iy_field"],
        t1w_affine  = ctx["t1w_affine"],
        t1w_header  = ctx["t1w_header"],
        mni_affine  = deform["mni_affine"],
        verbose     = verbose,
    )
    t_save     = time.time() - t0
    t_total    = time.time() - t_start

    # ── Summary stats ─────────────────────────────────────────────────
    import numpy as np
    n_gm  = int((posteriors[..., 0] > 0.5).sum())
    n_wm  = int((posteriors[..., 1] > 0.5).sum())
    n_csf = int((posteriors[..., 2] > 0.5).sum())

    if verbose:
        print("\n" + "=" * 60)
        print("  SEGMENTATION COMPLETE")
        print("=" * 60)
        print(f"  GM  voxels : {n_gm:>10,}")
        print(f"  WM  voxels : {n_wm:>10,}")
        print(f"  CSF voxels : {n_csf:>10,}")
        print()
        print(f"  Load       : {t_load:.1f}s")
        print(f"  Bias       : {t_bias:.1f}s")
        print(f"  GMM        : {t_gmm:.1f}s")
        print(f"  MRF        : {t_mrf:.1f}s")
        print(f"  Deform     : {t_deform:.1f}s")
        print(f"  Save       : {t_save:.1f}s")
        print(f"  Total      : {t_total:.1f}s")
        print()
        for k, v in output_files.items():
            print(f"  {k:4s} -> {Path(v).name}")
        print("=" * 60 + "\n")

    return {
        "output_files" : output_files,
        "time_load"    : round(t_load,   2),
        "time_bias"    : round(t_bias,   2),
        "time_gmm"     : round(t_gmm,    2),
        "time_mrf"     : round(t_mrf,    2),
        "time_deform"  : round(t_deform, 2),
        "time_save"    : round(t_save,   2),
        "time_total"   : round(t_total,  2),
        "n_gm_voxels"  : n_gm,
        "n_wm_voxels"  : n_wm,
        "n_csf_voxels" : n_csf,
    }


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="SPM-matched Segmentation"
    )
    p.add_argument("--t1w",      required=True,
                   help="T1w structural NIfTI  (T1w.nii)")
    p.add_argument("--tpm",      required=True,
                   help="SPM TPM.nii path  (<SPM_dir>/tpm/TPM.nii)")
    p.add_argument("--bias-reg", type=float, default=0.0001)
    p.add_argument("--bias-fwhm",type=float, default=60.0)
    p.add_argument("--mrf",      type=float, default=1.0,
                   help="MRF strength  (default: 1)")
    p.add_argument("--sampling", type=float, default=3.0,
                   help="Sampling distance mm  (default: 3)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_segmentation(
        t1w_path     = args.t1w,
        tpm_path     = args.tpm,
        bias_reg     = args.bias_reg,
        bias_fwhm    = args.bias_fwhm,
        mrf_strength = args.mrf,
        sampling_mm  = args.sampling,
    )