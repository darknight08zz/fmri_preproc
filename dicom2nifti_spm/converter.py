import os
import json
import numpy as np
import nibabel as nib
from .dicom_reader import read_dicom_directory, sort_slices
from .geometry import compute_affine


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


    slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))


    pos_map = {}

    for s in slices:
        pos_tuple = tuple([float(x) for x in s.ImagePositionPatient])
        pos_map.setdefault(pos_tuple, []).append(s)

    num_slices = len(pos_map)
    num_timepoints = len(slices) // num_slices

    print(f" Detected {num_slices} slices and {num_timepoints} timepoints.")

    first_slice = slices[0]
    rows = first_slice.Rows
    cols = first_slice.Columns


    image_data = np.zeros(
        (num_timepoints, num_slices, rows, cols),
        dtype=np.float32
    )

    unique_slices = [group[0] for group in pos_map.values()]
    sorted_unique_slices = sort_slices(unique_slices)

    pos_to_z = {
        tuple([float(x) for x in s.ImagePositionPatient]): i
        for i, s in enumerate(sorted_unique_slices)
    }


    slices.sort(
        key=lambda s: (
            float(getattr(s, "AcquisitionTime", 0)),
            int(getattr(s, "InstanceNumber", 0))
        )
    )

    z_time_counters = {z: 0 for z in range(num_slices)}


    for s in slices:

        pos_tuple = tuple([float(x) for x in s.ImagePositionPatient])
        z_idx = pos_to_z[pos_tuple]
        t_idx = z_time_counters[z_idx]

        if t_idx >= num_timepoints:
            continue

        pixel_data = s.pixel_array.astype(np.float32)

        slope = getattr(s, "RescaleSlope", 1.0)
        intercept = getattr(s, "RescaleIntercept", 0.0)

        pixel_data = pixel_data * float(slope) + float(intercept)

        image_data[t_idx, z_idx, :, :] = pixel_data
        z_time_counters[z_idx] += 1

    nifti_data = image_data.transpose(3, 2, 1, 0)

    if num_timepoints == 1:
        nifti_data = nifti_data.squeeze(3)


    affine = compute_affine(sorted_unique_slices)
    img = nib.Nifti1Image(nifti_data, affine)


    series_desc = getattr(first_slice, "SeriesDescription", "converted")
    series_num = getattr(first_slice, "SeriesNumber", "0")

    safe_desc = "".join(
        [c if c.isalnum() else "_" for c in str(series_desc)]
    )

    filename = f"series_{series_num}_{safe_desc}.nii"
    output_path = os.path.join(output_dir, filename)

    nib.save(img, output_path)
    print(f" Saved NIfTI: {output_path}")


    metadata = {}

    if hasattr(first_slice, "RepetitionTime"):
        metadata["RepetitionTime"] = float(first_slice.RepetitionTime) / 1000.0

    if hasattr(first_slice, "EchoTime"):
        metadata["EchoTime"] = float(first_slice.EchoTime)

    if hasattr(first_slice, "FlipAngle"):
        metadata["FlipAngle"] = float(first_slice.FlipAngle)

    if hasattr(first_slice, "Manufacturer"):
        metadata["Manufacturer"] = str(first_slice.Manufacturer).strip()

    if hasattr(first_slice, "MagneticFieldStrength"):
        metadata["MagneticFieldStrength"] = float(first_slice.MagneticFieldStrength)

    metadata["NumberOfSlices"] = num_slices
    metadata["NumberOfVolumes"] = num_timepoints

    try:
        slice_times = []

        acq_times = []

        for s in sorted_unique_slices:
            if hasattr(s, "AcquisitionTime"):
                acq_times.append(float(s.AcquisitionTime))

        if len(acq_times) == num_slices:

            base_time = min(acq_times)
            slice_times = [(t - base_time) for t in acq_times]

            metadata["SliceTiming"] = slice_times
            print(" SliceTiming successfully inferred from AcquisitionTime.")

    except Exception:
        print(" Could not infer SliceTiming from DICOM.")

    json_path = output_path.replace(".nii", ".json")

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f" Saved metadata JSON: {json_path}")
