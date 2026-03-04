
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates
from scipy.optimize import minimize

class MotionEstimator:
    """
    Estimates rigid-body motion parameters (6 degrees of freedom) 
    for 3D volumes relative to a reference volume.
    """
    
    def __init__(self, data, affine=None):
        """
        Initialize the estimator with 4D data.
        
        Args:
            data (numpy.ndarray): 4D image data (x, y, z, t).
            affine (numpy.ndarray, optional): 4x4 affine matrix. Used for grid generation.
        """
        if len(data.shape) != 4:
            raise ValueError("Data must be 4D (x, y, z, t)")
            
        self.data = data
        self.affine = affine if affine is not None else np.eye(4)
        self.dims = data.shape[:3]
        self.n_volumes = data.shape[3]
        
    def estimate_motion(self, ref_vol_idx=0):
        """
        Estimates motion parameters for all volumes relative to a reference volume.
        
        Args:
            ref_vol_idx (int): Index of the reference volume (default: 0).
            
        Returns:
            numpy.ndarray: (n_volumes, 6) array of motion parameters.
                           Columns: [tx, ty, tz, rx, ry, rz]
        """
        # 1. Select Reference Volume (I_ref)
        ref_vol = self.data[..., ref_vol_idx]
        
        # Pre-calculate coordinates for the reference volume
        # We optimize by using a grid of coordinates
        # Center of rotation is usually the center of the image
        img_center = np.array(self.dims) / 2.0
        
        # Coordinate grid (x, y, z)
        # We can use a mask here later to speed up (e.g., exclude background)
        # For simplicity, we use the whole grid for now, but mask out very low values
        mask = ref_vol > ref_vol.mean() * 0.1 # Simple threshold to ignore background
        coords = np.array(np.where(mask)) # (3, N_voxels)
        self.ref_values = ref_vol[mask]
        
        # Center the coordinates for rotation
        self.centered_coords = coords - img_center[:, np.newaxis]
        self.img_center = img_center
        
        # Initialize parameters [tx, ty, tz, rx, ry, rz]
        # rx, ry, rz in radians
        initial_params = np.zeros(6)
        
        motion_params = np.zeros((self.n_volumes, 6))
        
        print(f"Estimating motion for {self.n_volumes} volumes (Ref: {ref_vol_idx})...")
        
        for t in range(self.n_volumes):
            if t == ref_vol_idx:
                continue # Motion is 0 for reference
                
            moving_vol = self.data[..., t]
            
            # Optimization (Minimize SSD)
            # We use Powell or L-BFGS-B. SPM uses Gauss-Newton, but scipy's minimize is easier to implement robustly without analytic derivatives for now.
            # Powell is derivative-free and works well for this dimensionality.
            res = minimize(
                self._cost_function, 
                initial_params, 
                args=(moving_vol,),
                method='Powell',
                tol=1e-4,
                options={'maxiter': 100, 'disp': False}
            )
            
            motion_params[t, :] = res.x
            
            # Update initial guess for next volume (assuming smooth motion)
            initial_params = res.x
            
            if t % 10 == 0:
                print(f"  Vol {t}/{self.n_volumes}: {res.x}")
                
        return motion_params
        
    def _cost_function(self, params, moving_vol):
        """
        Computes the Sum of Squared Differences (SSD) between 
        the reference and the transformed moving volume.
        """
        tx, ty, tz, rx, ry, rz = params
        
        # 1. Build Transformation Matrix (Rigid Body)
        # Rotation matrix (Euler angles, ZYX order usually)
        # Small angle approximation is faster calculate but full rotation is more accurate
        # We use full rotation matrix here
        
        # Rotation matrices
        c, s = np.cos(rx), np.sin(rx)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        
        c, s = np.cos(ry), np.sin(ry)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        c, s = np.cos(rz), np.sin(rz)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Combined Rotation R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        
        # Translation vector
        T = np.array([tx, ty, tz])
        
        # 2. Transform Coordinates
        # New_coords = R * (Coords - Center) + Center + T
        # (Inverse mapping is usually used for interpolation: Source <- Target)
        # But optimize uses forward model usually?
        # Standard approach: Map Target(Ref) coordinates back to Source(Moving) to sample intensity
        # Coords_source = R^-1 * (Coords_target - Center - T) + Center
        
        # Inverse Rotation is Transpose for orthogonal matrices
        # Inverse Translation is -T
        # Let's verify the direction. We want to sample Moving Image at locations that correspond to Reference.
        # If Ref(x) corresponds to Mov(T(x)), we want the intensity at T(x) in Moving.
        # So we transform Ref coordinates BY the parameters to find where they land in Moving space.
        
        rotated_coords = R @ self.centered_coords
        transformed_coords = rotated_coords + self.img_center[:, np.newaxis] + T[:, np.newaxis]
        
        # 3. Interpolate (Trilinear)
        # map_coordinates uses spline interpolation (order 1 = linear)
        # Mode='nearest' or 'constant' to handle boundaries
        sampled_values = map_coordinates(
            moving_vol, 
            transformed_coords, 
            order=1, 
            mode='constant', 
            cval=0,
            prefilter=False
        )
        
        # 4. Compute Residual (SSD)
        # Simple Sum of Squared Differences
        # We can ignore zero values to avoid boundary artifacts dominating
        diff = sampled_values - self.ref_values
        ssd = np.sum(diff**2)
        
        return ssd

if __name__ == "__main__":
    # Test stub
    pass
