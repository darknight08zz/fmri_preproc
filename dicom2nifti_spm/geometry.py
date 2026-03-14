import numpy as np


def compute_affine(sorted_slices):
    """
    Compute the 4x4 RAS affine matrix for a NIfTI file from sorted DICOM slices.

    This function was correct in the original. Preserved with added comments
    explaining each step for clarity.

    Steps:
        1. Extract pixel spacing (dx, dy) and image orientation (IOP)
        2. Compute slice step vector from first/last slice positions
        3. Build 4x4 affine in DICOM's LPS coordinate system
        4. Convert LPS → RAS (NIfTI standard) by flipping X and Y
    """

    if not sorted_slices:
        raise ValueError("No slices provided for affine computation")

    first_slice = sorted_slices[0]
    last_slice  = sorted_slices[-1]

    # Pixel spacing: [row_spacing, col_spacing] in mm
    # row_spacing = distance between rows = step along column direction
    # col_spacing = distance between columns = step along row direction
    pixel_spacing = first_slice.PixelSpacing
    dy = float(pixel_spacing[0])   # row spacing → Y step
    dx = float(pixel_spacing[1])   # col spacing → X step

    # Image Orientation Patient: 6 floats
    # iop[:3] = direction cosines of the row direction (F→R or similar)
    # iop[3:] = direction cosines of the column direction
    iop      = [float(x) for x in first_slice.ImageOrientationPatient]
    r_vector = np.array(iop[:3])   # row direction cosine
    c_vector = np.array(iop[3:])   # column direction cosine

    # Normalize (DICOM spec says unit vectors, but defensive is good)
    r_vector = r_vector / np.linalg.norm(r_vector)
    c_vector = c_vector / np.linalg.norm(c_vector)

    # Slice normal = cross product of row and column cosines
    slice_normal = np.cross(r_vector, c_vector)

    # Slice step (dz): distance between consecutive slices
    n_slices = len(sorted_slices)
    if n_slices > 1:
        p0         = np.array([float(x) for x in first_slice.ImagePositionPatient])
        p1         = np.array([float(x) for x in last_slice.ImagePositionPatient])
        total_dist = np.dot((p1 - p0), slice_normal)
        dz         = total_dist / (n_slices - 1)   # (n-1) gaps between n slices ✓
    else:
        # Single-slice fallback: use SliceThickness if available
        dz = float(getattr(first_slice, 'SliceThickness', 1.0))

    # Build column vectors for affine (each = physical step per voxel)
    col_step_vec   = r_vector * dx          # step along X (columns)
    row_step_vec   = c_vector * dy          # step along Y (rows)
    slice_step_vec = slice_normal * dz      # step along Z (slices)

    # Origin = physical position of first voxel [0,0,0]
    origin = np.array([float(x) for x in first_slice.ImagePositionPatient])

    # Assemble affine in LPS (DICOM native coordinate system)
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = col_step_vec    # X column
    affine_lps[:3, 1] = row_step_vec    # Y column
    affine_lps[:3, 2] = slice_step_vec  # Z column
    affine_lps[:3, 3] = origin          # translation (origin)

    # Convert LPS → RAS (NIfTI standard)
    # LPS: Left, Posterior, Superior
    # RAS: Right, Anterior, Superior
    # Flip X (L→R) and Y (P→A): multiply rows 0 and 1 by -1
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])

    return lps_to_ras @ affine_lps