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
        Runs K-Means Segmentation (5 classes for robustness).
        Classes: 0=Background, 1=CSF, 2=GM, 3=WM, 4=Skull/Scalp.
        Produces:
            - output_base_pve_0.nii.gz (CSF)
            - output_base_pve_1.nii.gz (GM)
            - output_base_pve_2.nii.gz (WM)
            - output_base_restore.nii.gz (Bias Corrected - Mocked)
        """
        print(f"Running Native Segmentation (5-Class K-Means) on {input_path}")
        
        try:
            img = nib.load(input_path)
            data = img.get_fdata()
            affine = img.affine
            
            # 1. Robust Pre-masking (Otsu-like or simple)
            # Remove absolute zero background
            # Hist-based threshold to remove air
            hist, bins = np.histogram(data[data > 0], bins=100)
            # Otsu approx: data > mean? Or simple percentile?
            # Skull stripping is hard without bet.
            # Let's rely on clustering to separate "Dark Background" from tissues.
            
            # Flatten
            flat_data = data.reshape(-1, 1)
            
            # 2. K-Means with 5 classes
            # We expect: 
            # C0 (Darkest) = Background (Air)
            # C1 = CSF
            # C2 = GM
            # C3 = WM
            # C4 (Brightest) = Skull/Fat/Scalp
            
            print("  Clustering voxels (k=5)...")
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=5)
            # Subsample for speed if needed, but 5 classes 1mm^3 is fast enough usually
            # Or fit on non-zero?
            mask_nz = data > 0
            data_nz = data[mask_nz].reshape(-1, 1)
            
            if len(data_nz) < 1000:
                print("Error: Image seems empty.")
                return False, None
                
            kmeans.fit(data_nz)
            centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)
            
            # Map Cluster Index -> Sorted Rank (0..4)
            # rank_map[original_cluster_id] = rank
            rank_map = np.zeros(5, dtype=int)
            rank_map[sorted_indices] = np.arange(5)
            
            # Get labels for full image (predict 0s as well? no, 0s are 0)
            # Predict full image to get spatial map
            # This is heavy. Alternatively predict only nz and fill.
            labels_nz = kmeans.predict(data_nz)
            
            # Map labels to Rank (0=BG... 4=Skull)
            ranked_labels_nz = rank_map[labels_nz]
            
            # Create Volume of Ranks
            rank_vol = np.zeros_like(data, dtype=int)
            rank_vol[mask_nz] = ranked_labels_nz
            
            # 3. Probability Maps via Softmax/Distance (MRF-like)
            # We only want CSF (Rank 1), GM (Rank 2), WM (Rank 3).
            # We discard Rank 0 (BG) and Rank 4 (Skull).
            
            print("  Recovering tissue probabilities...")
            # Distance to centers
            sorted_centers = centers[sorted_indices]
            # dist shape: (N_voxels, 5)
            dists = kmeans.transform(data_nz)
            
            # Reorder dists columns to match Ranks 0..4
            # dists[:, i] is dist to center i. 
            # We want dists_ranked[:, 0] to be dist to center with Rank 0.
            # Center with Rank 0 is center[sorted_indices[0]]
            # So dists_ranked[:, k] = dists[:, sorted_indices[k]]
            dists_ranked = dists[:, sorted_indices]
            
            # Convert to Probabilities: Softmax(-dist^2 / T)
            # Temperature T controls sharpness.
            sigma = np.std(data_nz) * 0.5
            logits = - (dists_ranked ** 2) / (2 * sigma**2)
            
            # Softmax per voxel
            # exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            # probs_nz = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            # shape (N_nz, 5)
            
            # We can just output these probabilities? 
            # But let's apply Spatial Smoothing (MRF approximation).
            
            # Create 4D prob volume (X,Y,Z, 5)
            probs_vol = np.zeros(data.shape + (5,))
            
            # Optimize: do calculation on masked? Yes.
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs_nz = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Fill volume
            nz_indices = np.where(mask_nz)
            # This assignment is slow in Python loop. Advanced indexing?
            # probs_vol[mask_nz] = probs_nz # Works if broadcasting handles last dim?
            # mask_nz is (X,Y,Z). probs_vol is (X,Y,Z,5).
            # probs_vol[mask_nz, :] should work.
            probs_vol[mask_nz, :] = probs_nz
            
            # 4. Smoothing
            print("  smoothing probabilities...")
            for i in range(5):
                probs_vol[..., i] = scipy.ndimage.gaussian_filter(probs_vol[..., i], sigma=1.0)
                
            # Re-normalize sums?
            # probs_vol /= np.sum(probs_vol, axis=3, keepdims=True) + 1e-9
            
            # 5. Extract Tissues (1=CSF, 2=GM, 3=WM)
            # Save files
            tissues = [1, 2, 3] # Ranks
            out_maps = [] # For QC plot
            
            # Map Probabilities
            for i, rank in enumerate(tissues):
                # i=0 -> CSF (rank 1)
                p_map = probs_vol[..., rank]
                out_name = f"{output_base}_pve_{i}.nii.gz"
                nib.save(nib.Nifti1Image(p_map, affine, img.header), out_name)
                out_maps.append(p_map)
                
            # Bias Corrected (Mock: Multiply original by inverse variability? Hard without real field)
            # Just save original.
            nib.save(img, f"{output_base}_restore.nii.gz")
            
            # Generate QC Plot
            qc_plot_path = f"{output_base}_qc.png"
            self._generate_qc_plot(input_path, out_maps, sorted_indices, qc_plot_path)
            
            return True, qc_plot_path
            
        except Exception as e:
            print(f"Native Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def _generate_qc_plot(self, t1_path, maps, sorted_indices, out_path):
        try:
            import matplotlib.pyplot as plt
            
            t1_img = nib.load(t1_path)
            t1 = t1_img.get_fdata()
            
            # Use middle slices for 3 views to be robust
            x, y, z = t1.shape
            mx, my, mz = x // 2, y // 2, z // 2
            
            slices = [
                 (t1[mx, :, :].T, "Sagittal"),
                 (t1[:, my, :].T, "Coronal"),
                 (t1[:, :, mz].T, "Axial")
            ]
            
            # Maps: list of 3 ND arrays
            # We want to form an RGB image where:
            # R = GM, G = WM, B = CSF? Or standard:
            # Red=GM, Blue=WM, Green=CSF is requested in prompt.
            # maps[0]=CSF, maps[1]=GM, maps[2]=WM (as sorted in run())
            # Wait, `run` logic: "maps[0] corresponds to sorted_indices[0]"
            # In `run`: sorted_indices = argsort(centers). 
            # Darkest(0) -> Brightest(2).
            # T1: CSF(Dark) < GM < WM(Bright).
            # So: 0=CSF, 1=GM, 2=WM.
             
            csf_vol = maps[0]
            gm_vol = maps[1]
            wm_vol = maps[2]
            
            # Prepare RGB overlays for each slice
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            zooms = t1_img.header.get_zooms()[:3]
            aspects = [zooms[2]/zooms[1], zooms[2]/zooms[0], zooms[1]/zooms[0]]
            
            for i, (slice_t1, title) in enumerate(slices):
                ax = axes[i]
                
                # Normalize T1
                slice_t1_norm = (slice_t1 - np.min(slice_t1)) / (np.max(slice_t1) - np.min(slice_t1) + 1e-9)
                
                # Get map slices
                if i == 0:   # Sag
                    s_csf = csf_vol[mx, :, :].T
                    s_gm = gm_vol[mx, :, :].T
                    s_wm = wm_vol[mx, :, :].T
                elif i == 1: # Cor
                    s_csf = csf_vol[:, my, :].T
                    s_gm = gm_vol[:, my, :].T
                    s_wm = wm_vol[:, my, :].T
                else:        # Axi
                    s_csf = csf_vol[:, :, mz].T
                    s_gm = gm_vol[:, :, mz].T
                    s_wm = wm_vol[:, :, mz].T
                
                # Create RGB
                # R=GM, G=CSF, B=WM (Requested: GM=Red, WM=Blue, CSF=Green)
                rgb = np.zeros(slice_t1.shape + (3,))
                rgb[..., 0] = s_gm  # Red
                rgb[..., 1] = s_csf # Green
                rgb[..., 2] = s_wm  # Blue
                
                # Alpha depends on total probability? 
                # Or just additive blending?
                # We can overlay RGB on Grayscale.
                # If we use imshow(RGB), it replaces. We want alpha.
                # Construct RGBA
                
                # Combined probability
                total_prob = s_gm + s_csf + s_wm
                total_prob = np.clip(total_prob, 0, 1)
                
                # Normalize RGB by total prob to blend colors correctly?
                # Actually if GM=0.5, WM=0.5, we want purple.
                # Our simple assignment works.
                
                # Display T1
                ax.imshow(slice_t1_norm, cmap='gray', interpolation='nearest', aspect=aspects[i])
                
                # Display Overlay
                # Matplotlib imshow(RGBA)
                rgba = np.dstack((rgb, total_prob * 0.5)) # 0.5 opacity
                
                ax.imshow(rgba, interpolation='nearest', aspect=aspects[i])
                
                ax.set_title(f"{title}\nRed:GM Green:CSF Blue:WM")
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Seg QC Plot failed: {e}")
            import traceback
            traceback.print_exc()

        except Exception as e:
            print(f"Native Segmentation failed: {e}")
            # Fallback mock
            for i in range(3):
                 shutil.copy(input_path, f"{output_base}_pve_{i}.nii.gz")
            shutil.copy(input_path, f"{output_base}_restore.nii.gz")
            return False
