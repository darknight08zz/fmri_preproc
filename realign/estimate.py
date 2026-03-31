import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, affine_transform
from scipy.optimize import minimize


class MotionEstimator:
    """
    SPM-style rigid-body motion estimator (6 DOF).

    Two improvements over previous version:
    ─────────────────────────────────────────────────────────────
    IMPROVEMENT 1 — Two-pass alignment  (SPM spm_realign.m)
        Pass 1: align every volume to volume[0]
        Pass 2: build mean from pass-1 result,
                re-align every volume to that mean
        Why: mean image is more stable than any single volume.

    IMPROVEMENT 2 — World-space coordinates (mm, not voxels)
        Coordinates transformed via affine before optimization.
        Translations in rp_*.txt are now in mm (SPM standard).
        Why: for anisotropic voxels (e.g. 3.5×3.5×5mm), voxel
             translations ≠ mm and break motion QC thresholds.
    ─────────────────────────────────────────────────────────────
    """

    def __init__(self, data, affine=None):
        """
        Args:
            data   (np.ndarray): 4D image (X, Y, Z, T)
            affine (np.ndarray): 4×4 voxel-to-world matrix
        """
        if data.ndim != 4:
            raise ValueError("Data must be 4D (X, Y, Z, T)")

        self.data       = data.astype(np.float32)
        self.affine     = affine if affine is not None else np.eye(4)
        self.affine_inv = np.linalg.inv(self.affine)
        self.dims       = data.shape[:3]
        self.n_volumes  = data.shape[3]
        self.sigma      = 3.0    # ≈ 8mm FWHM (SPM default)
        self.sampling   = 3      # SPM style: only use every Nth voxel for speed

    # ──────────────────────────────────────────────────────────
    # PUBLIC: two-pass motion estimation
    # ──────────────────────────────────────────────────────────

    def estimate_motion(self):
        """
        Two-pass motion estimation — matches SPM spm_realign.m.

            Pass 1: align all volumes to volume[0]
            Pass 2: compute mean of pass-1 aligned data,
                    re-align all volumes to that mean

        Returns:
            np.ndarray: (N, 6) — [tx(mm), ty(mm), tz(mm),
                                   rx(rad), ry(rad), rz(rad)]
        """

        # ── PASS 1 ────────────────────────────────────────────
        print("[ESTIMATE] Pass 1: aligning to first volume...")
        ref_pass1    = self.data[..., 0].copy()
        params_pass1 = self._run_pass(ref_pass1, pass_num=1)

        # ── BUILD MEAN FROM PASS-1 RESULT ─────────────────────
        print("[ESTIMATE] Building mean image from pass-1 result...")
        mean_vol = self._build_mean(params_pass1)

        # ── PASS 2 ────────────────────────────────────────────
        print("[ESTIMATE] Pass 2: aligning to mean image...")
        params_pass2 = self._run_pass(mean_vol, pass_num=2)

        print("[ESTIMATE] Complete.")
        return params_pass2

    # ──────────────────────────────────────────────────────────
    # Single alignment pass
    # ──────────────────────────────────────────────────────────

    def _run_pass(self, ref_vol, pass_num=1):
        """
        Align all volumes to ref_vol.
        Returns (N, 6) parameter array.
        """
        # Smooth reference (SPM smooths at 8mm FWHM before cost function)
        ref_smooth = gaussian_filter(ref_vol.astype(np.float32), sigma=self.sigma)

        # Brain mask — SPM uses 0.8 * mean of smoothed image
        mask = ref_smooth > ref_smooth.mean() * 0.8

        # Apply spatial sampling for speed
        if self.sampling > 1:
            s_mask = np.zeros_like(mask)
            s_mask[::self.sampling, ::self.sampling, ::self.sampling] = True
            mask &= s_mask

        # ── IMPROVEMENT 2: voxel → world space (mm) ───────────
        #
        # Previous: used raw voxel indices as coordinates
        # Now:      transform each masked voxel to mm via affine
        #
        # Effect: translations in rp_ file are in mm (SPM standard)
        # For isotropic 3mm voxels: no numeric difference
        # For anisotropic (3.5×3.5×5mm): translations differ by
        # the ratio of voxel sizes — critical for motion QC.
        # ──────────────────────────────────────────────────────
        vox_idx     = np.array(np.where(mask), dtype=np.float64)   # (3, N)
        ones        = np.ones((1, vox_idx.shape[1]))
        world_coords = (self.affine @ np.vstack([vox_idx, ones]))[:3]   # mm

        # Rotation pivot = image center in mm
        center_vox   = np.array(self.dims, dtype=np.float64) / 2.0
        center_world = (self.affine @ np.append(center_vox, 1.0))[:3]

        # Store for cost function
        self._world_coords  = world_coords
        self._center_world  = center_world
        self._ref_values    = ref_smooth[mask].astype(np.float32)

        motion_params  = np.zeros((self.n_volumes, 6))
        initial_params = np.zeros(6)

        for t in range(self.n_volumes):

            moving_smooth = gaussian_filter(
                self.data[..., t].astype(np.float32), sigma=self.sigma
            )

            result = minimize(
                self._cost_function,
                initial_params,
                args=(moving_smooth,),
                method='Powell',
                tol=1e-5,
                options={'maxiter': 200, 'disp': False}
            )

            motion_params[t, :] = result.x
            initial_params      = result.x    # warm start for next volume

            if t % 10 == 0:
                p = result.x
                print(
                    f"  [Pass {pass_num}] Vol {t:3d}/{self.n_volumes} | "
                    f"t=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]mm  "
                    f"r=[{p[3]:.4f},{p[4]:.4f},{p[5]:.4f}]rad"
                )

        return motion_params

    # ──────────────────────────────────────────────────────────
    # Build mean image from pass-1 parameters
    # ──────────────────────────────────────────────────────────

    def _build_mean(self, motion_params):
        """
        Apply pass-1 transforms and average — produces the pass-2 reference.
        This is exactly what SPM does between its two alignment passes.
        """
        accumulator = np.zeros(self.dims, dtype=np.float64)
        center_vox  = np.array(self.dims, dtype=np.float64) / 2.0

        for t in range(self.n_volumes):
            p         = motion_params[t]
            R, _      = self._build_R(p[3], p[4], p[5])
            T         = p[:3]    # mm

            # Convert mm translation to voxel space for scipy
            # voxel offset = affine_inv applied to T
            T_vox = (self.affine_inv[:3, :3] @ T)

            offset = center_vox + T_vox - R @ center_vox

            aligned = affine_transform(
                self.data[..., t],
                matrix=R,
                offset=offset,
                order=1,
                mode='constant',
                cval=0.0
            )
            accumulator += aligned.astype(np.float64)

        mean_vol = (accumulator / self.n_volumes).astype(np.float32)
        print(f"[ESTIMATE] Mean image computed from {self.n_volumes} volumes.")
        return mean_vol

    # ──────────────────────────────────────────────────────────
    # Cost function — SSD in world space
    # ──────────────────────────────────────────────────────────

    def _cost_function(self, params, moving_vol):
        """
        SSD between reference and transformed moving volume.

        IMPROVEMENT 2 (world-space):
        - Rotation applied around center_world (mm)
        - Translation in mm
        - Transformed mm coords mapped back to voxels via affine_inv
          for interpolation
        """
        tx, ty, tz, rx, ry, rz = params
        R, _  = self._build_R(rx, ry, rz)
        T     = np.array([tx, ty, tz], dtype=np.float64)

        # Rotate around world center, then translate
        centered    = self._world_coords - self._center_world[:, np.newaxis]
        world_trans = R @ centered + self._center_world[:, np.newaxis] + T[:, np.newaxis]

        # World → voxel for interpolation
        ones     = np.ones((1, world_trans.shape[1]))
        vox_trans = (self.affine_inv @ np.vstack([world_trans, ones]))[:3]

        sampled = map_coordinates(
            moving_vol,
            vox_trans,
            order=1,
            mode='constant',
            cval=0.0,
            prefilter=False
        )

        diff = sampled.astype(np.float32) - self._ref_values
        return float(np.sum(diff ** 2))

    # ──────────────────────────────────────────────────────────
    # Helper: build 3x3 rotation matrix (XYZ order — SPM)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_R(rx, ry, rz):
        """
        Build 3×3 rotation matrix in SPM XYZ order: R = Rx @ Ry @ Rz
        Returns (R, None) — tuple for potential future Jacobian extension.
        """
        c, s = np.cos(rx), np.sin(rx)
        Rx   = np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

        c, s = np.cos(ry), np.sin(ry)
        Ry   = np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

        c, s = np.cos(rz), np.sin(rz)
        Rz   = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

        return Rx @ Ry @ Rz, None