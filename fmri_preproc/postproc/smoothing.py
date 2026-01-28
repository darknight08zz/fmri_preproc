
import nibabel as nib
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Union, Tuple

class SpatialSmoothing:
    """
    Apply spatial smoothing (Gaussian blur) to a NIfTI image.
    """
    
    def run(self, in_file: str, out_file: str, fwhm: float = 6.0) -> Tuple[str, str]:
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
        
        # Get voxel sizes (zoom) to adjust sigma per axis
        zooms = header.get_zooms()[:3] # (x, y, z)
        
        # Calculate sigma from FWHM
        # Sigma = FWHM / (2 * sqrt(2 * ln(2))) ~= FWHM / 2.35482
        
        if isinstance(fwhm, (list, tuple, np.ndarray)):
            if len(fwhm) != 3:
                # Handle scalar in list or expand?
                # For now assume 3-long list if list
                pass
            sigmas = [(f / 2.35482) / z for f, z in zip(fwhm, zooms)]
        else:
             sigma_mm = fwhm / 2.35482
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
        
        # Generate QC Plot
        qc_plot_path = out_file.replace(".nii.gz", "_qc.png").replace(".nii", "_qc.png")
        self._generate_qc_plot(in_file, out_file, qc_plot_path)
        
        return out_file, qc_plot_path

    def _generate_qc_plot(self, original_path, smoothed_path, out_path):
        try:
            import matplotlib.pyplot as plt
            
            orig = nib.load(original_path).get_fdata()
            smooth = nib.load(smoothed_path).get_fdata()
            
            # Middle slice/volume
            sl = orig.shape[2] // 2
            if len(orig.shape) == 4:
                vol = 0 
                orig_slice = orig[..., sl, vol]
                smooth_slice = smooth[..., sl, vol]
            else:
                orig_slice = orig[..., sl]
                smooth_slice = smooth[..., sl]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            ax1.imshow(orig_slice.T, cmap='gray', origin='lower')
            ax1.set_title("Original")
            ax1.axis('off')
            
            ax2.imshow(smooth_slice.T, cmap='gray', origin='lower')
            ax2.set_title("Smoothed")
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            
        except Exception as e:
            print(f"Smooth QC Plot failed: {e}")
