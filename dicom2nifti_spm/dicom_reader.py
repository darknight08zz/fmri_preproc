import os
import pydicom
import numpy as np
from collections import defaultdict


def read_dicom_directory(directory):
    """
    Scan directory recursively and group DICOM files by SeriesInstanceUID.

    FIX 1: Two-pass reading — headers only first, then pixel data.
    Original code used stop_before_pixels=False for every file just to
    read SeriesInstanceUID and check tags. For large fMRI datasets
    (hundreds of DICOM files), this loads gigabytes of pixel data
    unnecessarily during the grouping pass.

    FIX 2: SliceThickness removed from required_tags.
    SliceThickness (0050,0050) is an optional DICOM tag. It is absent
    in some scanner exports (e.g. Philips). geometry.py only uses it
    as a last-resort fallback for single-slice volumes, so rejecting
    valid multi-slice series because SliceThickness is missing is wrong.
    """

    print(f"Scanning {directory} for DICOM files...")

    # ----------------------------------------------------------
    # PASS 1: Read headers only (stop_before_pixels=True)
    # Group files by SeriesInstanceUID without loading pixel data.
    # ----------------------------------------------------------
    header_map = defaultdict(list)   # SeriesInstanceUID → list of file paths

    for root, _, files in os.walk(directory):
        for fname in files:
            file_path = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(
                    file_path,
                    stop_before_pixels=True,   # FIX: header only for grouping
                    force=True
                )

                if 'SeriesInstanceUID' not in ds:
                    continue

                # FIX: SliceThickness removed from required check
                required_tags = [
                    'ImagePositionPatient',
                    'ImageOrientationPatient',
                    'PixelSpacing',
                ]

                if not all(tag in ds for tag in required_tags):
                    continue

                header_map[ds.SeriesInstanceUID].append(file_path)

            except Exception:
                continue

    print(f"Found {len(header_map)} series.")

    # ----------------------------------------------------------
    # PASS 2: Re-read files WITH pixel data, now grouped by series.
    # Only loads pixel data for files that passed the header check.
    # ----------------------------------------------------------
    dicom_groups = defaultdict(list)

    for uid, file_paths in header_map.items():
        for file_path in file_paths:
            try:
                ds = pydicom.dcmread(
                    file_path,
                    stop_before_pixels=False,   # full read for pixel data
                    force=True
                )

                if 'PixelData' not in ds:
                    continue

                dicom_groups[uid].append(ds)

            except Exception:
                continue

    return dicom_groups


def sort_slices(slices):
    """
    Sort slices by their projection onto the slice-normal vector.
    This gives correct anatomical Z-ordering regardless of acquisition order.
    """
    if not slices:
        return []

    iop         = slices[0].ImageOrientationPatient
    row_cosines = np.array([float(x) for x in iop[:3]])
    col_cosines = np.array([float(x) for x in iop[3:]])

    # Slice normal = cross product of row and column direction cosines
    slice_normal = np.cross(row_cosines, col_cosines)

    slices.sort(
        key=lambda s: np.dot(
            np.array([float(x) for x in s.ImagePositionPatient]),
            slice_normal
        )
    )

    return slices