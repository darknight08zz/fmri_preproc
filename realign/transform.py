
import numpy as np

class TransformBuilder:
    """
    Constructs 4x4 affine transformation matrices from 
    rigid-body motion parameters.
    """
    
    @staticmethod
    def build_matrix(params):
        """
        Builds a 4x4 rigid-body transformation matrix from 6 parameters.
        
        Args:
            params (list or array): [tx, ty, tz, rx, ry, rz]
                - tx, ty, tz: Translation in mm
                - rx, ry, rz: Rotation in radians
        
        Returns:
            numpy.ndarray: 4x4 affine matrix
        """
        tx, ty, tz, rx, ry, rz = params
        
        # Rotation matrices
        c, s = np.cos(rx), np.sin(rx)
        Rx = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
        
        c, s = np.cos(ry), np.sin(ry)
        Ry = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
        
        c, s = np.cos(rz), np.sin(rz)
        Rz = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Combined Rotation
        R = Rz @ Ry @ Rx
        
        # Translation Matrix
        T = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        
        # Final Affine Matrix M = T * R
        # Order depends on convention: Rotate then Translate is standard
        M = T @ R
        
        return M

    @staticmethod
    def build_all_matrices(motion_params):
        """
        Builds matrices for all time points.
        
        Args:
            motion_params (numpy.ndarray): (N, 6) array of parameters.
            
        Returns:
            numpy.ndarray: (N, 4, 4) array of affine matrices.
        """
        n_vols = motion_params.shape[0]
        matrices = np.zeros((n_vols, 4, 4))
        
        for i in range(n_vols):
            matrices[i] = TransformBuilder.build_matrix(motion_params[i])
            
        return matrices
