import os
import shutil
import numpy as np
import nibabel as nib
import scipy.ndimage
from sklearn.cluster import KMeans

from typing import Tuple
class Segmentation:
    """
    Native Python Segmentation using K-Means Clustering.
    Replaces FSL FAST dependency.
    """
    def run(self, input_path: str, output_base: str) -> Tuple[bool, str]:
        """
        Runs K-Means Segmentation (3 classes).
        Produces:
            - output_base_pve_0.nii.gz (CSF)
            - output_base_pve_1.nii.gz (GM)
            - output_base_pve_2.nii.gz (WM)
            - output_base_restore.nii.gz (Bias Corrected - Mocked as original)
        
        Returns:
            Tuple[bool, str]: (success, qc_plot_path)
        """
        print(f"Running Native Segmentation (K-Means) on {input_path}")
        
        try:
            img = nib.load(input_path)
            data = img.get_fdata()
            affine = img.affine
            
            # Mask background (0s from skull stripping? Or robust range)
            mask = data > (np.mean(data) * 0.1) # Simple threshold
            masked_data = data[mask].reshape(-1, 1)
            
            if len(masked_data) == 0:
                print("Error: No data in mask.")
                return False
                
            # K-Means with 3 clusters (+ Background is 0)
            print("  Clustering voxels...")
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            # fit_predict on masked data
            labels_flat = kmeans.fit_predict(masked_data)
            centers = kmeans.cluster_centers_.flatten()
            
            # Sort centers by intensity: Darkest=CSF, Medium=GM, Brightest=WM
            sorted_indices = np.argsort(centers)
            # mapping: cluster_idx -> tissue_type (0=CSF, 1=GM, 2=WM)
            
            # --- MRF-like Regularization ---
            # Instead of just taking the raw labels (speckled), we calculate probabilities
            # and smooth them (simulating spatial prior).
            
            print("  Applying MRF-like Spatial Regularization...")
            
            # 1. Calculate squared distance to each center for all masked voxels
            # dist shape: (N_voxels, 3)
            # We can use transform() or manual
            dists = kmeans.transform(masked_data)
            
            # 2. Convert to probability-like weights: P ~ exp(-dist^2)
            # Normalize to avoid underflow? 
            # We care about relative values.
            # sigma estimated from variance? Let's fix a heuristic.
            sigma = np.std(masked_data) * 0.5
            probs_flat = np.exp(- (dists ** 2) / (2 * sigma**2))
            
            # Normalize rows to sum to 1
            row_sums = probs_flat.sum(axis=1, keepdims=True)
            probs_flat = probs_flat / (row_sums + 1e-9)
            
            # 3. Map back to volume for smoothing
            probs_vol = np.zeros(data.shape + (3,)) # (X,Y,Z,3)
            
            # We need full indices of masked data
            # mask is boolean volume
            mask_indices = np.where(mask)
            
            # Fill 3D volumes
            # We need to map sorted_indices to 0,1,2 standard order
            # dists[:, 0] corresponds to center[0], which might be WM.
            # We want probs_vol[..., 0] to be CSF (sorted_indices[0]).
            
            # Invert sorting: if sorted_indices is [2, 0, 1] (meaning center 2 is CSF...)
            # Then we want probs_vol[..., 0] = probs_flat[:, 2]
            
            for tissue_idx in range(3):
                # Which center corresponds to this tissue type?
                center_idx = sorted_indices[tissue_idx]
                probs_vol[mask_indices[0], mask_indices[1], mask_indices[2], tissue_idx] = probs_flat[:, center_idx]

            # 4. Spatial Smoothing (Regularization)
            # Smooth each probability map
            smoothed_probs = np.zeros_like(probs_vol)
            for i in range(3):
                # Sigma=1.0mm roughly (voxel size dependent usually, assume 1-2 voxels)
                smoothed_probs[..., i] = scipy.ndimage.gaussian_filter(probs_vol[..., i], sigma=1.0)
                
            # 5. Re-classify (Argmax) on smoothed probabilities
            # This cleans up speckles because a lone pixel of GM in WM will be overwhelmed by WM neighbors
            final_labels = np.argmax(smoothed_probs, axis=3)
            
            # Create output maps
            maps = [np.zeros_like(data) for _ in range(3)]
            
            for i in range(3):
                # Where mask is true AND final label is i
                # Note: background (mask=False) remains 0
                maps[i][(mask) & (final_labels == i)] = 1.0
            
            # Save files
            # pve_0 (CSF), pve_1 (GM), pve_2 (WM)
            for i in range(3):
                out_name = f"{output_base}_pve_{i}.nii.gz"
                nib.save(nib.Nifti1Image(maps[i], affine, img.header), out_name)
                
            # Save "Restored" (Bias corrected)
            restore_name = f"{output_base}_restore.nii.gz"
            nib.save(img, restore_name)
            
            # Generate QC Plot
            qc_plot_path = f"{output_base}_qc.png"
            self._generate_qc_plot(input_path, maps, sorted_indices, qc_plot_path)
            
            return True, qc_plot_path
            
        except Exception as e:
            print(f"Native Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback mock
            for i in range(3):
                 shutil.copy(input_path, f"{output_base}_pve_{i}.nii.gz")
            shutil.copy(input_path, f"{output_base}_restore.nii.gz")
            return False, None

    def _generate_qc_plot(self, t1_path, maps, sorted_indices, out_path):
        try:
            import matplotlib.pyplot as plt
            
            t1 = nib.load(t1_path).get_fdata()
            sl = t1.shape[2] // 2
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # T1
            axes[0].imshow(t1[..., sl].T, cmap='gray', origin='lower')
            axes[0].set_title("T1w")
            axes[0].axis('off')
            
            titles = ["CSF", "GM", "WM"]
            for i in range(3):
                # Map sorted index back to map list order? 
                # maps[0] corresponds to sorted_indices[0] (CSF), etc.
                tissue_map = maps[i]
                axes[i+1].imshow(tissue_map[..., sl].T, cmap='gray', origin='lower')
                axes[i+1].set_title(titles[i])
                axes[i+1].axis('off')
                
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
        except Exception as e:
            print(f"Seg QC Plot failed: {e}")

        except Exception as e:
            print(f"Native Segmentation failed: {e}")
            # Fallback mock
            for i in range(3):
                 shutil.copy(input_path, f"{output_base}_pve_{i}.nii.gz")
            shutil.copy(input_path, f"{output_base}_restore.nii.gz")
            return False
