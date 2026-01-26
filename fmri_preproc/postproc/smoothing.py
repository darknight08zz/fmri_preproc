
import nibabel as nib
import numpy as np
import scipy.ndimage
from typing import Union, Tuple

class SpatialSmoothing:
    """
    Apply spatial smoothing (Gaussian blur) to a NIfTI image.
    """
    
    def run(self, in_file: str, out_file: str, fwhm: float = 6.0) -> str:
        """
        Run spatial smoothing.
        
        Args:
            in_file: Path to input NIfTI file (3D or 4D).
            out_file: Path to save smoothed NIfTI file.
            fwhm: Full Width at Half Maximum (mm) for the Gaussian kernel.
            
        Returns:
             Path to output file.
        """
        print(f"Running Spatial Smoothing on {in_file} with FWHM={fwhm}mm")
        
        img = nib.load(in_file)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        
        # Calculate sigma from FWHM
        # Sigma = FWHM / (2 * sqrt(2 * ln(2))) ~= FWHM / 2.35482
        sigma_mm = fwhm / 2.35482
        
        # Get voxel sizes (zoom) to adjust sigma per axis
        zooms = header.get_zooms()[:3] # (x, y, z)
        
        # Sigma in voxels = Sigma_mm / Voxel_size_mm
        sigmas = [sigma_mm / s for s in zooms]
        
        # Handle 4D data (time series)
        if len(data.shape) == 4:
            # We only smooth spatially (first 3 dims), not temporally
            # scipy.ndimage.gaussian_filter takes a sequence of sigmas
            # We set sigma=0 for the 4th dimension (time)
            full_sigmas = sigmas + [0] 
        else:
            full_sigmas = sigmas
            
        smoothed_data = scipy.ndimage.gaussian_filter(data, sigma=full_sigmas)
        
        smoothed_img = nib.Nifti1Image(smoothed_data, affine, header)
        nib.save(smoothed_img, out_file)
        
        return out_file
