import numpy as np
import pandas as pd
from typing import Tuple

class MotionMetrics:
    """
    Calculates Framewise Displacement (FD) and DVARS.
    """
    def calculate_fd(self, motion_params_file: str) -> np.ndarray:
        """
        Calculates FD from 6 motion parameters (3 trans, 3 rot).
        Assumes rotational params in radians.
        Converts rotations to mm using 50mm radius.
        """
        # Load params (assuming format from FSL: 3 rot, 3 trans or vice versa)
        # FSL mcflirt: rotx roty rotz tx ty tz (radians, mm)
        params = np.loadtxt(motion_params_file)
        
        # Derivatives
        diff = np.diff(params, axis=0)
        diff = np.vstack((np.zeros(6), diff)) # pad first timepoint
        
        # Convert rotations to displacement (radius = 50mm)
        # Assuming cols 0-2 are rotation
        diff[:, 0:3] = diff[:, 0:3] * 50.0 
        
        fd = np.sum(np.abs(diff), axis=1)
        return fd

    def calculate_dvars(self, func_path: str, mask_path: str = None) -> np.ndarray:
        """
        Calculates DVARS (RMS change in BOLD signal).
        """
        import nibabel as nib
        img = nib.load(func_path)
        data = img.get_fdata()
        
        # Calculate global DVARS
        # Diff along time axis
        diff_data = np.diff(data, axis=-1)
        
        # Helper: RMS over space
        def rms(arr):
            return np.sqrt(np.mean(np.square(arr)))
            
        dvars = [0] # first point
        # This is slow in pure python loop, but okay for prototype
        # Correct vectorization:
        # square -> mean over space -> sqrt
        
        # Reshape to (voxels, time)
        n_voxels = np.prod(data.shape[:3])
        flat_diff = diff_data.reshape(n_voxels, -1)
        
        # RMS over voxels for each timepoint
        dvars_ts = np.sqrt(np.mean(np.square(flat_diff), axis=0))
        dvars.extend(dvars_ts)
        
        return np.array(dvars)
