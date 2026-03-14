import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt


class Reslicer:
    """
    Applies estimated motion parameters to reslice 4D fMRI data,
    and generates mean image and motion QA plots.

    The reslice logic (offset formula, scipy call) was correct in the
    original. Minor improvements added:
    - float32 enforced on output to save memory
    - Interpolation order exposed as parameter (default=1, trilinear)
    - Cleaner comments on the inverse-mapping logic
    """

    def __init__(self, data, affine, header, output_paths):
        """
        Args:
            data         (np.ndarray): 4D data (X, Y, Z, T)
            affine       (np.ndarray): 4x4 affine matrix
            header                   : NIfTI header
            output_paths (dict)      : paths from VolumeLoader.get_output_filenames()
        """
        self.data         = data.astype(np.float32)
        self.affine       = affine
        self.header       = header
        self.output_paths = output_paths
        self.dims         = data.shape[:3]
        self.n_volumes    = data.shape[3]

    # ----------------------------------------------------------
    # Reslice
    # ----------------------------------------------------------

    def reslice(self, motion_matrices, interp_order=1):
        """
        Reslice volumes using estimated transformation matrices.

        Inverse mapping logic:
            estimate.py defined the FORWARD transform:
                coords_moving = R @ (coords_ref - center) + center + T
                             = R @ coords_ref + (center + T - R @ center)

            scipy.ndimage.affine_transform uses:
                output[o] = input[matrix @ o + offset]

            So matrix = R  and  offset = center + T - R @ center
            This is EXACTLY the inverse mapping (Output=Ref → Input=Moving),
            which is what we need for correct interpolation.

        Args:
            motion_matrices (np.ndarray): (N, 4, 4) array of affine matrices
            interp_order    (int)       : 1=trilinear (default), 3=cubic

        Returns:
            np.ndarray: Resliced 4D data (float32)
        """
        print("[RESLICE] Reslicing volumes...")
        resliced_data = np.zeros_like(self.data, dtype=np.float32)

        center = np.array(self.dims, dtype=np.float64) / 2.0

        for t in range(self.n_volumes):

            if t % 10 == 0:
                print(f"  Volume {t:3d}/{self.n_volumes}")

            M = motion_matrices[t]
            R = M[:3, :3].astype(np.float64)
            T = M[:3,  3].astype(np.float64)   # [tx, ty, tz]

            # Offset for scipy inverse mapping:
            # output[o] = input[R @ o + offset]
            # where offset = center + T - R @ center
            offset = center + T - R @ center

            resliced_data[..., t] = affine_transform(
                self.data[..., t],
                matrix=R,
                offset=offset,
                order=interp_order,
                mode='constant',
                cval=0.0,
                prefilter=(interp_order > 1)
            ).astype(np.float32)

        return resliced_data

    # ----------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------

    def save_outputs(self, resliced_data, motion_params):
        """
        Save resliced 4D NIfTI, mean image, motion parameters (.txt),
        and motion QA plot (.png).
        """
        print("[RESLICE] Saving outputs...")

        # 1. Resliced 4D
        resliced_img = nib.Nifti1Image(
            resliced_data.astype(np.float32), self.affine, self.header
        )
        nib.save(resliced_img, self.output_paths['resliced'])
        print(f"  Resliced 4D  : {self.output_paths['resliced']}")

        # 2. Mean image (computed from resliced data — SPM behavior)
        mean_data = np.mean(resliced_data, axis=3).astype(np.float32)
        mean_img  = nib.Nifti1Image(mean_data, self.affine, self.header)
        nib.save(mean_img, self.output_paths['mean'])
        print(f"  Mean image   : {self.output_paths['mean']}")

        # 3. Motion parameters (.txt) — SPM rp_ format
        np.savetxt(self.output_paths['motion_params'], motion_params, fmt='%.6f')
        print(f"  Motion params: {self.output_paths['motion_params']}")

        # 4. Motion QA plot
        self._plot_motion(motion_params)

    # ----------------------------------------------------------
    # Motion QA plot
    # ----------------------------------------------------------

    def _plot_motion(self, params):
        """
        SPM-style motion plot: translations (mm) and rotations (rad).
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # Translations
        axes[0].plot(params[:, :3])
        axes[0].set_title('Translation (mm)')
        axes[0].legend(['x', 'y', 'z'])
        axes[0].set_xlabel('Volume')
        axes[0].grid(True)

        # Rotations
        axes[1].plot(params[:, 3:])
        axes[1].set_title('Rotation (radians)')
        axes[1].legend(['pitch (rx)', 'roll (ry)', 'yaw (rz)'])
        axes[1].set_xlabel('Volume')
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = str(self.output_paths['motion_params']).replace('.txt', '.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Motion plot  : {plot_path}")