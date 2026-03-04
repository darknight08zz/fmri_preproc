import numpy as np

def compute_affine(sorted_slices):
    if not sorted_slices:
        raise ValueError("No slices provided for affine computation")

    first_slice = sorted_slices[0]
    last_slice = sorted_slices[-1]
    
    pixel_spacing = first_slice.PixelSpacing
    dy, dx = float(pixel_spacing[0]), float(pixel_spacing[1])
    
    iop = [float(x) for x in first_slice.ImageOrientationPatient]
    r_vector = np.array(iop[:3])
    c_vector = np.array(iop[3:])
    
    r_vector = r_vector / np.linalg.norm(r_vector)
    c_vector = c_vector / np.linalg.norm(c_vector)
    
    slice_normal = np.cross(r_vector, c_vector)
    
    n_slices = len(sorted_slices)
    if n_slices > 1:
        p0 = np.array([float(x) for x in first_slice.ImagePositionPatient])
        p1 = np.array([float(x) for x in last_slice.ImagePositionPatient])
        total_dist = np.dot((p1 - p0), slice_normal)
        dz = total_dist / (n_slices - 1)
    else:
        dz = float(first_slice.SliceThickness)
        
    col_step_vec = r_vector * dx
    row_step_vec = c_vector * dy
    slice_step_vec = slice_normal * dz
    
    origin = np.array([float(x) for x in first_slice.ImagePositionPatient])
    
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = col_step_vec
    affine_lps[:3, 1] = row_step_vec
    affine_lps[:3, 2] = slice_step_vec
    affine_lps[:3, 3] = origin
    
    lps_to_ras = np.diag([-1, -1, 1, 1])
    
    return np.dot(lps_to_ras, affine_lps)
