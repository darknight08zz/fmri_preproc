import numpy as np
import nibabel as nib

class Scaling:
    """
    Scaling to Percent Signal Change (PSC) or Z-score.
    """
    def run(self, input_path: str, output_path: str, method: str = 'psc') -> bool:
        print(f"Scaling {input_path} using method: {method}")
        try:
            img = nib.load(input_path)
            data = img.get_fdata()
            
            # Mean over time
            mean_img = np.mean(data, axis=-1) # (X, Y, Z)
            
            # Broadcast mean to matching dims
            # data is (X, Y, Z, T), mean_img is (X, Y, Z)
            # We need to divide data[..., t] by mean_img
            
            # Mask out background where mean is near zero
            mask = np.abs(mean_img) > 1e-6
            
            scaled_data = np.zeros_like(data)
            
            if method == 'psc':
                # (x - mean) / mean * 100 
                # = (x / mean - 1) * 100
                
                # Expand dims for broadcasting
                # We iterate or use fancy indexing
                # Let's use masked assignment
                
                # Careful with shape broadcasting
                mean_expanded = mean_img[..., np.newaxis]
                
                # Valid mask broadcasted
                # Actually, easier to compute difference
                # centered = data - mean_expanded
                # psc = (centered / mean_expanded) * 100
                
                # Safe division
                with np.errstate(divide='ignore', invalid='ignore'):
                     scaled_data = (data / mean_expanded - 1) * 100
                
                # Zero out bad voxels
                scaled_data[~mask] = 0
                scaled_data = np.nan_to_num(scaled_data)

            elif method == 'zscore':
                std_img = np.std(data, axis=-1)
                std_expanded = std_img[..., np.newaxis]
                mean_expanded = mean_img[..., np.newaxis]
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    scaled_data = (data - mean_expanded) / std_expanded
                
                scaled_data[~mask] = 0
                scaled_data = np.nan_to_num(scaled_data)
            
            nib.save(nib.Nifti1Image(scaled_data, img.affine, img.header), output_path)
            return True
        except Exception as e:
            print(f"Scaling failed: {e}")
            return False
