import os
import json
import numpy as np
import nibabel as nib
from .dicom_reader import read_dicom_directory, sort_slices
from .geometry import compute_affine


# ----------------------------------------------------------
# Helper: parse DICOM AcquisitionTime string → seconds
# ----------------------------------------------------------

def _parse_acq_time(acq_time_val) -> float:
    """
    Parse DICOM AcquisitionTime into seconds since midnight.

    DICOM AcquisitionTime format: HHMMSS.ffffff  (string)
    e.g. '120530.500' = 12h 05m 30.5s = 43530.5 seconds

    FIX: Original code cast directly to float(), which gives 120530.5
    instead of 43530.5 — a meaningless number that produces completely
    wrong SliceTiming differences and wrong volume sort order.
    """
    t = str(acq_time_val).strip()

    # Pad if needed (some scanners omit fractional seconds)
    if '.' not in t:
        t = t + '.000000'

    h  = int(t[0:2])
    m  = int(t[2:4])
    s  = float(t[4:])

    return h * 3600.0 + m * 60.0 + s


# ----------------------------------------------------------

def convert_directory(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    series_dict = read_dicom_directory(input_dir)

    for uid, slices in series_dict.items():
        try:
            print(f"\nProcessing series {uid} ({len(slices)} DICOM files)...")
            convert_series(slices, output_dir)
        except Exception as e:
            print(f"Failed to convert series {uid}: {e}")


def convert_series(slices, output_dir):

    if not slices:
        return

    # Initial sort by InstanceNumber
    slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))

    # Group slices by spatial position (Z-position fingerprint)
    pos_map = {}
    for s in slices:
        pos_tuple = tuple(float(x) for x in s.ImagePositionPatient)
        pos_map.setdefault(pos_tuple, []).append(s)

    num_slices     = len(pos_map)
    num_timepoints = len(slices) // num_slices

    print(f"  Detected {num_slices} slices and {num_timepoints} timepoints.")

    first_slice = slices[0]
    rows = first_slice.Rows
    cols = first_slice.Columns

    # Allocate: (T, Z, rows, cols)
    image_data = np.zeros(
        (num_timepoints, num_slices, rows, cols),
        dtype=np.float32
    )

    # Sort unique slice positions along slice normal
    unique_slices        = [group[0] for group in pos_map.values()]
    sorted_unique_slices = sort_slices(unique_slices)

    pos_to_z = {
        tuple(float(x) for x in s.ImagePositionPatient): i
        for i, s in enumerate(sorted_unique_slices)
    }

    # ----------------------------------------------------------
    # FIX 1: Sort all slices by AcquisitionTime using correct
    # HHMMSS.ffffff parser, not raw float() cast.
    #
    # Original: float(getattr(s, 'AcquisitionTime', 0))
    #   '120530.500' → 120530.5  (wrong — not seconds)
    #
    # Correct:  _parse_acq_time('120530.500') → 43530.5 seconds
    #
    # Wrong parsing means volumes are sorted incorrectly, corrupting
    # the 4D timeseries ordering.
    # ----------------------------------------------------------
    def sort_key(s):
        raw_t = getattr(s, 'AcquisitionTime', None)
        t_sec = _parse_acq_time(raw_t) if raw_t is not None else 0.0
        return (t_sec, int(getattr(s, 'InstanceNumber', 0)))

    slices.sort(key=sort_key)

    # Fill volume array
    z_time_counters = {z: 0 for z in range(num_slices)}

    for s in slices:
        pos_tuple = tuple(float(x) for x in s.ImagePositionPatient)
        z_idx = pos_to_z[pos_tuple]
        t_idx = z_time_counters[z_idx]

        if t_idx >= num_timepoints:
            continue

        pixel_data = s.pixel_array.astype(np.float32)
        slope      = float(getattr(s, 'RescaleSlope',     1.0))
        intercept  = float(getattr(s, 'RescaleIntercept', 0.0))
        pixel_data = pixel_data * slope + intercept

        image_data[t_idx, z_idx, :, :] = pixel_data
        z_time_counters[z_idx] += 1

    # Transpose (T, Z, rows, cols) → (cols, rows, Z, T) = (X, Y, Z, T)
    nifti_data = image_data.transpose(3, 2, 1, 0)

    if num_timepoints == 1:
        nifti_data = nifti_data.squeeze(3)

    # Build affine and save NIfTI
    affine = compute_affine(sorted_unique_slices)
    img    = nib.Nifti1Image(nifti_data, affine)

    series_desc = getattr(first_slice, 'SeriesDescription', 'converted')
    series_num  = getattr(first_slice, 'SeriesNumber',      '0')
    safe_desc   = ''.join(c if c.isalnum() else '_' for c in str(series_desc))

    filename    = f"series_{series_num}_{safe_desc}.nii"
    output_path = os.path.join(output_dir, filename)

    nib.save(img, output_path)
    print(f"  Saved NIfTI: {output_path}")

    # ----------------------------------------------------------
    # Build BIDS-style metadata JSON
    # ----------------------------------------------------------
    metadata = {}

    if hasattr(first_slice, 'RepetitionTime'):
        # DICOM TR is in milliseconds → convert to seconds for BIDS
        metadata['RepetitionTime'] = float(first_slice.RepetitionTime) / 1000.0

    if hasattr(first_slice, 'EchoTime'):
        # ----------------------------------------------------------
        # FIX 2: EchoTime must also be divided by 1000.
        # DICOM stores TE in milliseconds. BIDS spec requires seconds.
        # Original code stored raw ms value, breaking any downstream
        # tool that reads TE from the JSON (e.g., fieldmap correction).
        # ----------------------------------------------------------
        metadata['EchoTime'] = float(first_slice.EchoTime) / 1000.0

    if hasattr(first_slice, 'FlipAngle'):
        metadata['FlipAngle'] = float(first_slice.FlipAngle)   # degrees — no conversion

    if hasattr(first_slice, 'Manufacturer'):
        metadata['Manufacturer'] = str(first_slice.Manufacturer).strip()

    if hasattr(first_slice, 'MagneticFieldStrength'):
        metadata['MagneticFieldStrength'] = float(first_slice.MagneticFieldStrength)

    metadata['NumberOfSlices']  = num_slices
    metadata['NumberOfVolumes'] = num_timepoints

    # ----------------------------------------------------------
    # FIX 3: SliceTiming — parse AcquisitionTime correctly
    #
    # Original: acq_times.append(float(s.AcquisitionTime))
    # e.g. '120530.500' → 120530.5  (wrong unit, wrong differences)
    #
    # Correct: _parse_acq_time() → seconds since midnight
    # Differences between slice times are then real inter-slice gaps
    # (typically 0.05–0.1s), which is what SPM's STC expects.
    # ----------------------------------------------------------
    try:
        acq_times = []

        for s in sorted_unique_slices:
            if hasattr(s, 'AcquisitionTime'):
                acq_times.append(_parse_acq_time(s.AcquisitionTime))

        if len(acq_times) == num_slices:
            base_time  = min(acq_times)
            slice_times = [t - base_time for t in acq_times]
            metadata['SliceTiming'] = slice_times
            print("  SliceTiming successfully inferred from AcquisitionTime.")
        else:
            print("  Could not infer SliceTiming (incomplete AcquisitionTime tags).")

    except Exception as e:
        print(f"  Could not infer SliceTiming: {e}")

    json_path = output_path.replace('.nii', '.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"  Saved metadata JSON: {json_path}")