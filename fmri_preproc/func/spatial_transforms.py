import os
import shutil
import glob
import numpy as np
import nibabel as nib
import scipy.ndimage
from typing import List

class SpatialTransforms:
    """
    Native Python Spatial Transforms (Application).
    Replaces ANTs ApplyTransforms.
    """
    def run(self, input_path: str, reference_path: str, output_path: str, transforms: List[str]) -> bool:
        """
        Applies a list of transforms to an input image.
        Supports:
            - Static Affine Matrix (.mat / .txt output from NativeNormalization)
            - (Future: Warp Fields)
        """
        print(f"Running Native Spatial Transforms on {input_path}")
        
        try:
            # Load Input
            img = nib.load(input_path)
            data = img.get_fdata()
            affine = img.affine
            
            # Load Reference (Template)
            if not os.path.exists(reference_path):
                # Fallback to input geometry if template missing (should capture in Norm Node but double check)
                ref_shape = data.shape[:3]
                ref_affine = affine
            else:
                ref_img = nib.load(reference_path)
                ref_shape = ref_img.shape[:3]
                ref_affine = ref_img.affine
            
            # Combine Transforms
            # We might have [.mat (Affine)] and [.nii.gz (Warp)]
            # ANTs order: -t Warp -t Affine.
            # Means: Warp( Affine(x) ).
            # 1. Compute Affine Matrix (Composition of all .mats)
            
            affine_matrix = np.eye(4)
            warp_field = None # (X,Y,Z,3)
            
            # Iterate Reverse? ANTs applies Right to Left?
            # User passes [mat, warp].
            # Normalization.run returns [mat, warp].
            # Logic: Input -> Affine -> [Affine Aligned] -> Warp -> Output.
            
            for t_path in transforms:
                 if t_path.endswith(".mat") or t_path.endswith(".txt"):
                     try:
                         mat = np.loadtxt(t_path)
                         # Compose: New @ Current
                         affine_matrix = mat @ affine_matrix
                     except: pass
                     
                 elif t_path.endswith(".nii.gz") or t_path.endswith(".nii"):
                     # Load Warp
                     print(f"  Loading Warp Field: {t_path}")
                     w_img = nib.load(t_path)
                     warp_field = w_img.get_fdata() 
             
            # Prepare for Resampling
            # We want to map Output Grid (MNI) -> Input Grid (Func).
            # Our Warp Field (Demons output) is defined in MNI space (Ref space).
            # D(x_mni) points to x_affine.
            # So: x_affine = x_mni + D(x_mni).
            # And: x_input = InverseAffine(x_affine).
            # So: x_input = InvAffine( x_mni + D(x_mni) ).
            
            # 1. Create Output Grid (MNI coords)
            coords = np.meshgrid(
                np.arange(ref_shape[0]), 
                np.arange(ref_shape[1]), 
                np.arange(ref_shape[2]), 
                indexing='ij'
            )
            
            # 2. Add Warp (if exists)
            if warp_field is not None:
                # warp_field is (X,Y,Z,3)
                # coords is list of 3 (X,Y,Z) arrays
                # Add displacement
                # Note: Check displacement units (voxels vs mm). Demons used voxels (grid units).
                coords[0] = coords[0] + warp_field[..., 0]
                coords[1] = coords[1] + warp_field[..., 1]
                coords[2] = coords[2] + warp_field[..., 2]
                
            # Now `coords` contains coordinates in "Affine Space".
            
            # 3. Apply Inverted Affine to get "Input Space" coordinates
            # coords is (3, X, Y, Z) (logically)
            # Flatten for matrix mult
            flat_coords = np.vstack([c.flatten() for c in coords]) # (3, N)
            flat_coords = np.vstack([flat_coords, np.ones((1, flat_coords.shape[1]))]) # (4, N)
            
            # Invert Affine
            try:
                inv_msg = np.linalg.inv(affine_matrix)
            except:
                inv_msg = np.eye(4)
            
            # Map Affine -> Input
            input_coords_flat = inv_msg @ flat_coords # (4, N)
            
            # Reshape back for map_coordinates
            reshaped_coords = [
                input_coords_flat[0, :].reshape(ref_shape),
                input_coords_flat[1, :].reshape(ref_shape),
                input_coords_flat[2, :].reshape(ref_shape)
            ]
            
            # 4. Resample
            if len(data.shape) == 4:
                n_vols = data.shape[3]
                print(f"Applying Transforms (Affine+Warp) to {n_vols} volumes...")
                new_data = np.zeros(ref_shape + (n_vols,))
                for i in range(n_vols):
                    vol = data[..., i]
                    new_data[..., i] = scipy.ndimage.map_coordinates(
                        vol, reshaped_coords, order=1
                    )
            else:
                 new_data = scipy.ndimage.map_coordinates(
                    data, reshaped_coords, order=1
                )
            
            # Save
            new_img = nib.Nifti1Image(new_data, ref_affine, ref_img.header)
            nib.save(new_img, output_path)
            
            return True

        except Exception as e:
            print(f"Native Spatial Transforms failed: {e}")
            import traceback
            traceback.print_exc()
            shutil.copy(input_path, output_path)
            return False

        except Exception as e:
            print(f"Native Spatial Transforms failed: {e}")
            shutil.copy(input_path, output_path)
            return False
