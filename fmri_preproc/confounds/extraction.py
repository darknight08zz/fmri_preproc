import nibabel as nib
import numpy as np
import os
from typing import Dict


class SignalExtraction:
    """
    Extracts mean signals from masks (WM, CSF, Global).
    """
    def run(self, func_path: str, masks: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Args:
            masks: Dict like {"WM": "path/to/wm_mask.nii.gz", "CSF": "..."}
        """
        img = nib.load(func_path)
        data = img.get_fdata()
        
        signals = {}
        
        for name, mask_path in masks.items():
            if not os.path.exists(mask_path):
                print(f"Mask not found: {mask_path}")
                continue
                
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata() > 0.5 # binary
            
            # Ensure dims match
            if mask_data.shape != data.shape[:3]:
                print(f"Shape mismatch for {name}")
                continue
                
            # Extract mean
            # data[mask_data] returns (N_voxels_in_mask, Time)
            masked_ts = data[mask_data]
            mean_ts = np.mean(masked_ts, axis=0)
            signals[name] = mean_ts
            
        return signals
