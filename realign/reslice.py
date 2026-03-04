
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt

class Reslicer:
    """
    Applies estimated motion parameters to reslice 4D fMRI data
    and generates mean images and motion plots.
    """
    
    def __init__(self, data, affine, header, output_paths):
        """
        Initialize the reslicer.
        
        Args:
            data (numpy.ndarray): Original 4D data.
            affine (numpy.ndarray): Original affine matrix.
            header (nibabel.nifti1.Nifti1Header): Original header.
            output_paths (dict): Dictionary of output paths from Loader.
        """
        self.data = data
        self.affine = affine
        self.header = header
        self.output_paths = output_paths
        self.dims = data.shape[:3]
        self.n_volumes = data.shape[3]
        
    def reslice(self, motion_matrices):
        """
        Reslices the data using the estimated transformation matrices.
        
        Args:
            motion_matrices (numpy.ndarray): (N, 4, 4) array of affine matrices.
            
        Returns:
            numpy.ndarray: Resliced 4D data.
        """
        print("Reslicing volumes...")
        resliced_data = np.zeros_like(self.data)
        
        # Grid for interpolation
        # In reality, we want to map Target -> Source
        # If Matrix M maps Source -> Target (Ref), then we need M^-1 to pull from Source
        # Standard approach:
        #   New_Voxel_Coord = inv(M) * Old_Voxel_Coord
        
        for t in range(self.n_volumes):
            if t % 10 == 0:
                print(f"  Reslicing volume {t}/{self.n_volumes}...")
                
            M = motion_matrices[t]
            
            # Inverse of the estimated transform to pull data from original volume
            # The estimated params map Ref -> Moving (or vice-versa depending on cost function definition)
            # In estimate.py we minimized SSD(Ref, Transformed(Moving))
            # So params define Moving -> Ref
            # affine_transform uses the inverse convention (Output -> Input) automatically if we provide matrix?
            # Scipy affine_transform(input, matrix) defines: output[i] = input[matrix @ i]
            # So if M maps Moving -> Ref (Target), and we want to fill Target, we need Mapping Target -> Moving
            # Which is indeed inv(M).
            # BUT scipy.ndimage.affine_transform expects the INVERSE mapping by default (or forward if we interpret differently?)
            # Doc: "The matrix M maps the output coordinates to the input coordinates."
            # So we need M that maps Ref (Output) -> Moving (Input).
            # Our estimated M maps Ref -> Moving in estimate.py logic ( R * coords + T )?
            # Wait, in estimate.py: transformed_coords = R @ coords + T
            # This mapped Reference Grid -> Moving Space. 
            # So the matrix constructed from params IS the mapping Output(Ref) -> Input(Moving).
            # So we can pass it directly to affine_transform!
            
            # However, we must account for the center of rotation used in estimate.py
            # Center of rotation was image center.
            # Scipy affine_transform rotates around (0,0,0) by default.
            # We need to compose: Translate(-Center) -> Rotate -> Translate(Center)
            # Or just use the already computed transform if it accounts for it?
            # In estimate.py output: R @ (x - c) + c + T
            # = R@x - R@c + c + T
            # = R@x + (c + T - R@c)
            # So the effective offset is (c + T - R@c).
            
            # Let's reconstruct the full affine for Scipy
            center = np.array(self.dims) / 2.0
            
            # Extract 3x3 rotation and translation from 4x4 M
            R = M[:3, :3]
            T = M[:3, 3] # This T is from build_matrix, which is just translation
            
            # The offset required for scipy:
            offset = center + T - R @ center
            
            resliced_data[..., t] = affine_transform(
                self.data[..., t],
                matrix=M[:3, :3], # Rotation part
                offset=offset,    # Offset part
                order=1 # Linear interpolation (Trilinear) - efficient and standard
            )
            
        return resliced_data
        
    def save_outputs(self, resliced_data, motion_params):
        """
        Saves the resliced 4D file, mean image, and motion parameters.
        """
        print("Saving outputs...")
        
        # 1. Save Resliced 4D
        resliced_img = nib.Nifti1Image(resliced_data, self.affine, self.header)
        nib.save(resliced_img, self.output_paths['resliced'])
        print(f"  Saved resliced data: {self.output_paths['resliced']}")
        
        # 2. Save Mean Image
        mean_data = np.mean(resliced_data, axis=3)
        mean_img = nib.Nifti1Image(mean_data, self.affine, self.header)
        nib.save(mean_img, self.output_paths['mean'])
        print(f"  Saved mean image: {self.output_paths['mean']}")
        
        # 3. Save Motion Parameters
        np.savetxt(self.output_paths['motion_params'], motion_params, fmt='%.6f')
        print(f"  Saved motion parameters: {self.output_paths['motion_params']}")
        
        # 4. Plot Motion (Bonus QA)
        self._plot_motion(motion_params)
        
    def _plot_motion(self, params):
        """
        Generates a motion plot similar to SPM/FSL.
        """
        plt.figure(figsize=(10, 6))
        
        # Translations (first 3 cols)
        plt.subplot(2, 1, 1)
        plt.plot(params[:, :3])
        plt.title("Translation (mm)")
        plt.legend(['x', 'y', 'z'])
        plt.grid(True)
        
        # Rotations (last 3 cols)
        plt.subplot(2, 1, 2)
        plt.plot(params[:, 3:])
        plt.title("Rotation (radians)")
        plt.legend(['pitch', 'roll', 'yaw'])
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = str(self.output_paths['motion_params']).replace('.txt', '.png')
        plt.savefig(plot_path)
        print(f"  Saved motion plot: {plot_path}")
