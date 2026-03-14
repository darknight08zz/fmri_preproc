import numpy as np


class TransformBuilder:
    """
    Constructs 4x4 affine transformation matrices from
    rigid-body motion parameters.

    Fix applied vs original:
    - Rotation order corrected: Rx @ Ry @ Rz  (SPM XYZ order, was ZYX)
      MUST match the order used in estimate.py cost function.
    """

    @staticmethod
    def build_matrix(params):
        """
        Builds a 4x4 rigid-body transformation matrix from 6 parameters.

        Args:
            params: [tx, ty, tz, rx, ry, rz]
                    translations in mm, rotations in radians

        Returns:
            np.ndarray: 4x4 affine matrix
        """
        tx, ty, tz, rx, ry, rz = params

        # ----------------------------------------------------------
        # FIX: Rotation order = Rx @ Ry @ Rz  (SPM XYZ order)
        #
        # Original code used Rz @ Ry @ Rx (ZYX).
        # This MUST match the rotation order in estimate.py.
        # If they differ, the resliced output will be wrong because
        # the matrix passed to Reslicer encodes a different rotation
        # than the one that was optimized during estimation.
        # ----------------------------------------------------------

        c, s = np.cos(rx), np.sin(rx)
        Rx = np.array([
            [1,  0,  0,  0],
            [0,  c, -s,  0],
            [0,  s,  c,  0],
            [0,  0,  0,  1]
        ])

        c, s = np.cos(ry), np.sin(ry)
        Ry = np.array([
            [ c,  0,  s,  0],
            [ 0,  1,  0,  0],
            [-s,  0,  c,  0],
            [ 0,  0,  0,  1]
        ])

        c, s = np.cos(rz), np.sin(rz)
        Rz = np.array([
            [c, -s,  0,  0],
            [s,  c,  0,  0],
            [0,  0,  1,  0],
            [0,  0,  0,  1]
        ])

        # SPM rotation order: XYZ
        R = Rx @ Ry @ Rz

        # Translation matrix
        T = np.array([
            [1,  0,  0,  tx],
            [0,  1,  0,  ty],
            [0,  0,  1,  tz],
            [0,  0,  0,   1]
        ])

        # Final: rotate then translate (SPM convention)
        M = T @ R

        return M

    @staticmethod
    def build_all_matrices(motion_params):
        """
        Builds 4x4 matrices for all time points.

        Args:
            motion_params (np.ndarray): (N, 6) array

        Returns:
            np.ndarray: (N, 4, 4) array of affine matrices
        """
        n_vols   = motion_params.shape[0]
        matrices = np.zeros((n_vols, 4, 4))

        for i in range(n_vols):
            matrices[i] = TransformBuilder.build_matrix(motion_params[i])

        return matrices