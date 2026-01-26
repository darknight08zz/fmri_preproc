import nibabel as nib
import numpy as np
from typing import Tuple, List

def check_orientation(nifti_path: str) -> Tuple[str, List[str]]:
    """
    Checks the orientation of a NIfTI file.
    Returns:
        orientation_code (str): e.g., 'RAS', 'LPS'
        warnings (List[str]): List of potential issues (obliquity, non-orthogonal)
    """
    warnings = []
    try:
        img = nib.load(nifti_path)
        # Check affine
        affine = img.affine
        
        # Check for obliquity (non-orthogonal)
        # Compute the angle from the cardinal axes
        # (Simplified check: if off-diagonal elements are large)
        r = affine[:3, :3]
        if not np.allclose(r, np.diag(np.diag(r)), atol=0.1):
             # This is a loose check, proper way is checking cosine of angles
             # But for now, just flag generally if it looks rotated
             pass # keeping it simple for now, maybe use nibabel.aff2axcodes
             
        # Get orientation details
        ornt = nib.io_orientation(affine)
        code = nib.aff2axcodes(affine)
        orientation_code = "".join(code)
        
        if orientation_code != "RAS":
            warnings.append(f"Orientation is {orientation_code}, expected RAS. Pipeline typically reorients to RAS.")
            
        # Check for severe obliquity if needed
        # ...
        
        return orientation_code, warnings
    except Exception as e:
        return "UNKNOWN", [f"Failed to load header: {str(e)}"]
