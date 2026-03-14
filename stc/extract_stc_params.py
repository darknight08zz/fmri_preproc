"""
═══════════════════════════════════════════════════════════════
  STC PARAMETER EXTRACTOR — ADNI DATA
  Extracts: TR, nSlices, TA, SliceOrder, RefSlice
  Sources : NIfTI header + BIDS _bold.json sidecar
═══════════════════════════════════════════════════════════════

Usage:
    python extract_stc_params.py

Edit the two paths at the bottom before running.
"""

import os
import json
import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Run: pip install nibabel")


# ══════════════════════════════════════════════════════════════
#  MAIN EXTRACTOR
# ══════════════════════════════════════════════════════════════

def extract_stc_params(nifti_path: str, json_path: str = None):
    """
    Extract all 5 STC parameters from your ADNI NIfTI + JSON.

    Parameters
    ----------
    nifti_path : str
        Path to your 4D functional NIfTI.
        e.g. "sub-01/ses-01/func/sub-01_ses-01_task-rest_bold.nii"

    json_path : str or None
        Path to BIDS sidecar JSON.
        If None, auto-detected by replacing .nii with .json

    Returns
    -------
    dict with keys: TR, nSlices, nVolumes, TA, sliceOrder, refSlice
    """

    print("═" * 60)
    print("  STC PARAMETER EXTRACTOR")
    print("═" * 60)

    # ──────────────────────────────────────────────
    # AUTO-DETECT JSON IF NOT PROVIDED
    # ──────────────────────────────────────────────
    if json_path is None:
        json_path = nifti_path.replace(".nii.gz", ".json").replace(".nii", ".json")
        print(f"\n[JSON] Auto-detected: {json_path}")

    # ──────────────────────────────────────────────
    # STEP 1: LOAD NIfTI HEADER
    # Source of: nSlices, nVolumes, TR (backup)
    # ──────────────────────────────────────────────
    print(f"\n[NIfTI] Loading: {nifti_path}")

    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(f"NIfTI not found: {nifti_path}")

    img    = nib.load(nifti_path)
    header = img.header
    shape  = img.shape   # (X, Y, Z, T)

    print(f"[NIfTI] Shape : {shape}")

    # ── Number of Slices ──────────────────────────
    # Location: NIfTI header, 3rd spatial dimension (Z)
    # shape[2] = number of slices along Z axis
    nSlices = shape[2]
    print(f"\n{'─'*60}")
    print(f"PARAMETER 1 — Number of Slices")
    print(f"  Source : NIfTI header shape[2] (Z dimension)")
    print(f"  Value  : {nSlices}")

    # ── Number of Volumes ─────────────────────────
    nVolumes = shape[3] if len(shape) == 4 else 1
    print(f"\n  (nVolumes = {nVolumes}, for reference)")

    # ── TR from NIfTI header (backup source) ──────
    # Location: pixdim[4] = voxel size along 4th dimension = TR in seconds
    try:
        tr_nifti = float(header.get_zooms()[3])
    except Exception:
        tr_nifti = 0.0

    print(f"\n{'─'*60}")
    print(f"PARAMETER 2 — TR (Repetition Time)")
    print(f"  Backup source : NIfTI pixdim[4] = {tr_nifti:.4f} s")

    # ──────────────────────────────────────────────
    # STEP 2: LOAD BIDS JSON SIDECAR
    # Primary source of: TR, SliceTiming
    # ──────────────────────────────────────────────
    TR          = None
    sliceTiming = None

    if os.path.isfile(json_path):
        print(f"\n[JSON] Found sidecar: {json_path}")
        with open(json_path, "r") as f:
            bold_json = json.load(f)

        # ── TR from JSON (most reliable) ──────────
        # Location: _bold.json → "RepetitionTime" key
        # Unit: seconds (BIDS spec)
        if "RepetitionTime" in bold_json:
            TR = float(bold_json["RepetitionTime"])
            print(f"  RepetitionTime found in JSON: {TR} s")
        else:
            print(f"  WARNING: RepetitionTime not in JSON")

        # ── SliceTiming from JSON ─────────────────
        # Location: _bold.json → "SliceTiming" key
        # Format  : array of floats, one per slice
        #           each value = acquisition time of that slice in seconds
        # Example : [0.0, 0.5, 0.0625, 0.5625, 0.125, ...]
        #            ↑ slice 0 acquired at t=0.0s
        #                  ↑ slice 1 acquired at t=0.5s (interleaved!)
        if "SliceTiming" in bold_json:
            sliceTiming = bold_json["SliceTiming"]
            print(f"  SliceTiming found: {len(sliceTiming)} entries")
            print(f"  First 6 values: {[round(x,4) for x in sliceTiming[:6]]}")
        else:
            print(f"  WARNING: SliceTiming not in JSON")
            print(f"  → Will fall back to interleaved ascending assumption")

        # Show all JSON keys found (for reference)
        print(f"\n  All JSON keys found: {list(bold_json.keys())}")

    else:
        print(f"\n[JSON] NOT FOUND at {json_path}")
        print(f"  → Using NIfTI header for TR")
        print(f"  → SliceTiming will be assumed (interleaved ascending)")

    # ── Resolve TR ────────────────────────────────
    if TR is None:
        if tr_nifti > 0:
            TR = tr_nifti
            print(f"\n  TR resolved from NIfTI header: {TR:.4f} s")
        else:
            # Hardcoded ADNI fallback — change if your data differs
            TR = 2.5
            print(f"\n  WARNING: TR not found anywhere.")
            print(f"  Using ADNI default: {TR} s")
            print(f"  → Check your scanner protocol sheet to confirm!")

    print(f"\n  ✓ TR = {TR:.4f} s")

    # ──────────────────────────────────────────────
    # STEP 3: CALCULATE TA
    # TA is NEVER stored — always calculated
    # ──────────────────────────────────────────────
    # Formula: TA = TR - (TR / nSlices)
    # Meaning: time to acquire all slices, excluding the gap between volumes
    #
    # Example: TR=2.5s, nSlices=40
    #   Gap between volumes = TR/nSlices = 2.5/40 = 0.0625s
    #   TA = 2.5 - 0.0625 = 2.4375s
    TA = TR - (TR / nSlices)

    print(f"\n{'─'*60}")
    print(f"PARAMETER 3 — TA (Time of Acquisition)")
    print(f"  Source  : CALCULATED (never stored in any file)")
    print(f"  Formula : TR - (TR / nSlices)")
    print(f"  = {TR} - ({TR} / {nSlices})")
    print(f"  = {TR} - {TR/nSlices:.6f}")
    print(f"  ✓ TA = {TA:.6f} s")

    # ──────────────────────────────────────────────
    # STEP 4: DERIVE SLICE ORDER
    # ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"PARAMETER 4 — Slice Order")

    if sliceTiming is not None:
        # ── Method A: From JSON SliceTiming (most accurate) ──
        # SliceTiming[i] = time at which slice i was acquired
        # argsort gives us the order slices were acquired in
        #
        # Example SliceTiming = [0.0, 0.5, 0.0625, 0.5625]
        # argsort → [0, 2, 1, 3]  (slice 0 first, slice 2 second, etc.)
        sliceOrder = (np.argsort(sliceTiming) + 1).tolist()  # +1 for 1-based (SPM)
        print(f"  Source : _bold.json SliceTiming → argsort")
        print(f"  Method : sort slice indices by acquisition time")
        print(f"  ✓ Slice Order (first 10): {sliceOrder[:10]} ...")
        print(f"  (1-based indexing for SPM)")
    else:
        # ── Method B: Assume interleaved ascending (Siemens default) ──
        # ADNI data is mostly Siemens 3T (Prisma/Trio/Skyra)
        # Siemens default = interleaved ascending
        # Pattern: odd slices first, then even
        # e.g. nSlices=6 → [1, 3, 5, 2, 4, 6]
        odd  = list(range(1, nSlices + 1, 2))   # [1, 3, 5, ...]
        even = list(range(2, nSlices + 1, 2))   # [2, 4, 6, ...]
        sliceOrder = odd + even
        print(f"  Source : ASSUMED (no SliceTiming in JSON)")
        print(f"  Pattern: interleaved ascending (Siemens 3T default)")
        print(f"  ✓ Slice Order (first 10): {sliceOrder[:10]} ...")
        print(f"  ⚠ Verify this against your scanner protocol!")

    # Reference slice = anatomically middle slice (SPM default)
    # Formula: round(nSlices / 2)  →  1-based (SPM convention)
    # Python form (0-based): round(nSlices / 2) - 1
    refSlice = round(nSlices / 2)   # 1-based, as SPM expects

    print(f"\nPARAMETER 5 — Reference Slice")
    print(f"  Formula : round(nSlices / 2) = round({nSlices} / 2)")
    print(f"  Value   : {refSlice}  (1-based, SPM convention)")
    print(f"  0-based : {refSlice - 1}  (for Python SliceTimer)")

    # ──────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────
    params = {
        "TR":         TR,
        "nSlices":    nSlices,
        "nVolumes":   nVolumes,
        "TA":         TA,
        "sliceOrder": sliceOrder,
        "refSlice":   refSlice,
    }

    print(f"\n{'═'*60}")
    print(f"  FINAL STC PARAMETERS — READY TO USE")
    print(f"{'═'*60}")
    print(f"  TR         = {TR:.4f} s")
    print(f"  nSlices    = {nSlices}")
    print(f"  nVolumes   = {nVolumes}")
    print(f"  TA         = {TA:.6f} s")
    print(f"  sliceOrder = {sliceOrder[:8]} ...")
    print(f"  refSlice   = {refSlice}")
    print(f"{'═'*60}\n")

    return params


# ══════════════════════════════════════════════════════════════
#  HOW TO USE IN YOUR STC PIPELINE
# ══════════════════════════════════════════════════════════════

def show_usage_example(params: dict):
    """Shows how to plug extracted params into SPM matlabbatch."""
    print("HOW TO USE IN MATLAB (SPM matlabbatch):")
    print("─" * 60)
    print(f"""
matlabbatch{{1}}.spm.temporal.st.scans    = {{funcFile}};
matlabbatch{{1}}.spm.temporal.st.nslices  = {params['nSlices']};
matlabbatch{{1}}.spm.temporal.st.tr       = {params['TR']};
matlabbatch{{1}}.spm.temporal.st.ta       = {params['TA']:.6f};
matlabbatch{{1}}.spm.temporal.st.so       = {params['sliceOrder'][:6]}...;
matlabbatch{{1}}.spm.temporal.st.refslice = {params['refSlice']};
""")
    print("HOW TO USE IN PYTHON (our SliceTimer class):")
    print("─" * 60)
    print(f"""
from correction import SliceTimer
import nibabel as nib
import numpy as np

img  = nib.load("arfunc_4D.nii")   # after realignment
data = img.get_fdata(dtype="float32")

timer = SliceTimer(
    data        = data,
    tr          = {params['TR']},
    slice_order = {params['sliceOrder'][:6]}...,  # full list
    ref_slice   = {params['refSlice'] - 1},        # 0-based for Python
    ta          = {params['TA']:.6f},
)
corrected = timer.correct()
""")


# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── EDIT THESE TWO PATHS FOR YOUR ADNI DATA ──────────────
    NIFTI_PATH = "sub-01/ses-01/func/sub-01_ses-01_task-rest_bold.nii"
    JSON_PATH  = "sub-01/ses-01/func/sub-01_ses-01_task-rest_bold.json"
    # ─────────────────────────────────────────────────────────

    params = extract_stc_params(NIFTI_PATH, JSON_PATH)
    show_usage_example(params)