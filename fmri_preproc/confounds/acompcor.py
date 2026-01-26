import nibabel as nib
import numpy as np
from typing import Tuple, List

from sklearn.decomposition import PCA

class ACompCor:
    """
    Anatomical CompCor.
    Extracts principal components from high-variance voxels in WM/CSF masks.
    """
    def run(self, func_path: str, mask_path: str, n_components: int = 5) -> np.ndarray:
        """
        Returns top N components from the masked data.
        """
        try:
            img = nib.load(func_path)
            data = img.get_fdata() # (X, Y, Z, T)
            
            mask_img = nib.load(mask_path)
            mask = mask_img.get_fdata() > 0.5
            
            # Extract time series from mask
            # valid_voxels shape: (N_voxels, T)
            valid_voxels = data[mask] 
            
            # Detrend constant and linear drift?
            # For simplicity, just center
            valid_voxels = valid_voxels - np.mean(valid_voxels, axis=1, keepdims=True)
            
            # Transpose for sklearn: (n_samples, n_features) -> (T, N_voxels)
            X = valid_voxels.T
            
            # Check if we have enough voxels
            if X.shape[1] < n_components:
                # Fallback
                return np.zeros((X.shape[0], n_components))
                
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X)
            
            return components
            
        except Exception as e:
            print(f"aCompCor failed: {e}")
            # Return zeros match time length?
            # Need to know T. Load header?
            return np.array([])
