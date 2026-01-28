import os
import shutil
import numpy as np
import nibabel as nib
import scipy.ndimage
import scipy.optimize
from typing import Tuple, List

class Normalization:
    """
    Native Python Normalization (Affine Registration to MNI).
    Replaces ANTs Registration.
    """
    def __init__(self, template_path: str = "templates/MNI152_T1_2mm.nii.gz"):
        self.template_path = template_path

    def run(self, input_path: str, output_base: str) -> Tuple[str, List[str], str]:
        """
        Runs Affine Registration (12 DOF) to Template.
        Returns: (warped_image_path, list_of_transforms, qc_plot_path)
        """
        print(f"Running Native Normalization (Affine) on {input_path}")
        
        # Check template
        if not os.path.exists(self.template_path):
            print(f"Warning: Template {self.template_path} not found.")
            # Auto-create dummy template if missing (Simulated MNI Box)
            # OR just copy input to create "Self-Template" for demo?
            # Creating a dummy box is better to show "different" space.
            self._create_dummy_template()
            
        try:
             # Load Images
            ref_img = nib.load(self.template_path)
            mov_img = nib.load(input_path)
            
            ref_data = ref_img.get_fdata()
            mov_data = mov_img.get_fdata()
            
            # Normalize Intensities
            ref_data = self._normalize_intensity(ref_data)
            mov_data = self._normalize_intensity(mov_data)
            
            # Optimization (12 DOF: Scale(3), Shear(3), Rot(3), Trans(3))
            
            # 0. Center of Mass Initialization
            print("  Aligning Centers of Mass...")
            ref_com = scipy.ndimage.center_of_mass(ref_data)
            mov_com = scipy.ndimage.center_of_mass(mov_data)
            translation_init = np.array(ref_com) - np.array(mov_com)
            # COMs are in index space. Assuming same voxel size/orientation for simplicity of this native impl.
            # Ideally use affine to map to world. 
            # If native normalization assumes resampled input, indices might work.
            # But let's assume index alignment is a good start.
            
            # Params: [Sx, Sy, Sz, Shxy, Shxz, Shyz, Rx, Ry, Rz, Tx, Ty, Tz]
            # Initialize T with COM shift
            initial_params = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, translation_init[0], translation_init[1], translation_init[2]])
            
            print(f"Optimizing Affine Transform (12 DOF)...")
            
            # Powell is slow for 12 params. 
            # Improvement: Multi-stage? Rigid then Affine?
            # For now: Increase iterations and tolerance.
            res = scipy.optimize.minimize(
                self._nmi_loss_function,
                initial_params,
                args=(ref_data, mov_data),
                method='Powell',
                tol=1e-2, # Stricter tolerance
                options={'maxiter': 50, 'disp': True} # Increased from 5 to 50
            )
            
            best_params = res.x
            print(f"  Best Params: {best_params}, Loss={res.fun:.4f}")
            
            # Construct Matrix
            mat = self._params_to_matrix(best_params)
            
            # Save Matrix
            mat_path = output_base + "_0GenericAffine.mat"
            np.savetxt(mat_path, mat)
            
            # --- Non-Linear Demons Registration ---
            print(f"Running Fast Demons (Non-Linear) Registration...")
            
            # 1. Apply Affine to get starting point
            mov_img_orig = nib.load(input_path)
            # We need the full volume in memory
            affine_aligned = self._apply_transform(mov_img_orig.get_fdata(), best_params, ref_shape=ref_data.shape)
            
            # 2. Run Demons Loop
            # Returns (warped_data, displacement_field_4d)
            warped_data, disp_field = self._run_demons_registration(affine_aligned, ref_data)
            
            # 3. Save Warped Output
            out_warped = output_base + "_Warped.nii.gz"
            new_img = nib.Nifti1Image(warped_data, ref_img.affine, ref_img.header)
            nib.save(new_img, out_warped)
            
            # 4. Save Displacement Field (Warp)
            # Shape (X,Y,Z,3) - Vector field
            warp_path = output_base + "_1Warp.nii.gz"
            # We normally save ANTs style? Or native? 
            # Vector field: 5th dim in NIfTI is usually for vectors? Or 4th dim?
            # 4th dim is Time/Vector. 3 components.
            warp_img = nib.Nifti1Image(disp_field, ref_img.affine, ref_img.header)
            nib.save(warp_img, warp_path)
            
            # Returns transform list: Affine AND Warp
            # Generate QC Plot
            qc_plot_path = output_base + "_qc.png"
            self._generate_qc_plot(self.template_path, out_warped, qc_plot_path)
            
            return out_warped, [mat_path, warp_path], qc_plot_path # Order matters! Affine then Warp? Or Warp is composition?
            # Usuaully: Total(x) = Warp(Affine(x)).
            # Our Affine creates "Affine_Aligned". Demons creates Warp from "Affine_Aligned" -> Ref.
            # So T_total = Warp o Affine.
            # SpatialTransforms needs to handle this.
            
        except Exception as e:
            print(f"Native Normalization failed: {e}")
            import traceback
            traceback.print_exc()
            out_warped = output_base + "_error.nii.gz"
            shutil.copy(input_path, out_warped)
            # Dummy mat
            mat_path = output_base + "_0GenericAffine.mat"
            np.savetxt(mat_path, np.eye(4))
            return out_warped, [mat_path], None

    def _run_demons_registration(self, mov, ref, iterations=15, sigma_fluid=1.0):
        """
        Simple Demons Algorithm.
        mov, ref: 3D numpy arrays (normalized intensities).
        
        Returns:
            warped: 3D array
            field: 4D array (X,Y,Z,3) containing displacement vectors (u,v,w)
        """
        # Initialize Displacement Field (Zero)
        disp = np.zeros(mov.shape + (3,), dtype=np.float32)
        
        # Grid coordinates
        coords = np.meshgrid(
            np.arange(mov.shape[0]), 
            np.arange(mov.shape[1]), 
            np.arange(mov.shape[2]), 
            indexing='ij'
        )
        
        current_warped = mov.copy()
        
        # Iteration
        for i in range(iterations):
            # 1. Compute Difference
            diff = ref - current_warped
            
            # 2. Compute Gradient of Moving Image (at current warped position)
            # (Use Ref gradient for symmetric demons, but simple demons uses Moving)
            # Actually Thirion's original used Moving.
            grad = np.array(np.gradient(current_warped)) # shape (3, X, Y, Z)
            
            # 3. Calculate Force (Thirion's)
            # u = (diff * grad) / (norm(grad)^2 + diff^2)
            grad_norm2 = np.sum(grad**2, axis=0)
            denominator = grad_norm2 + diff**2
            denominator[denominator == 0] = 1e-9 # Avoid div zero
            
            # Update vector
            # shape (3, X,Y,Z)
            # diff is (X,Y,Z)
            delta_field = np.zeros_like(grad)
            for d in range(3):
                delta_field[d] = (diff * grad[d]) / denominator
                
            # 4. Regularization (Fluid smoothing of update)
            # Smooth the update field
            for d in range(3):
                delta_field[d] = scipy.ndimage.gaussian_filter(delta_field[d], sigma=sigma_fluid)
            
            # update field (Accumulate)
            # disp is (X,Y,Z,3) -> (3, X,Y,Z) for internal math
            # Small step approach: Disp_new = Disp_old + Delta
            # This is additive demons (assuming small deformation).
            # Compositive is better but additive is easier.
            
            # Move delta indices to last
            delta_field = np.moveaxis(delta_field, 0, -1)
            disp += delta_field
            
            # 5. Apply new field to warp Moving
            # Map = Identity + Disp
            # coords[0] + disp[...,0], etc.
            map_coords = [coords[d] + disp[..., d] for d in range(3)]
            
            # We map Output -> Input?
            # Demons calculates displacement u such that M(x+u) = F(x).
            # map_coordinates(input, coordinates) evaluates input at coordinates.
            # If we want value at x in Warped, we look at x+u in Moving.
            # So yes: Warped(x) = Moving(x + u).
            
            # Note: map_coordinates expects (3, N) or list of arrays
            current_warped = scipy.ndimage.map_coordinates(mov, map_coords, order=1)
            
            # MSE
            mse = np.mean((ref - current_warped)**2)
            if i % 5 == 0:
                print(f"    Demons Iter {i}: MSE={mse:.4f}")
                
        return current_warped, disp

    def _generate_qc_plot(self, tpl_path, warped_path, out_path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            tpl_img = nib.load(tpl_path)
            warped_img = nib.load(warped_path)
            
            tpl_data = tpl_img.get_fdata()
            warped_data = warped_img.get_fdata()
            
            # Use middle slices for 3 views to be robust
            x, y, z = tpl_data.shape
            mx, my, mz = x // 2, y // 2, z // 2
            
            # Use Template geometry for slicing
            slices_tpl = [
                tpl_data[mx, :, :].T,
                tpl_data[:, my, :].T,
                tpl_data[:, :, mz].T
            ]
            
            # Warped data should match template space
            slices_warped = [
                warped_data[mx, :, :].T,
                warped_data[:, my, :].T,
                warped_data[:, :, mz].T
            ]
            
            # Aspect ratios
            zooms = tpl_img.header.get_zooms()[:3]
            aspects = [
                zooms[2]/zooms[1], # Sag: Z/Y
                zooms[2]/zooms[0], # Cor: Z/X
                zooms[1]/zooms[0]  # Axi: Y/X
            ]
            
            titles = ["Sagittal", "Coronal", "Axial"]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, ax in enumerate(axes):
                # Background: Template (Gray)
                tpl_slice = slices_tpl[i]
                tpl_slice = (tpl_slice - np.min(tpl_slice)) / (np.max(tpl_slice) - np.min(tpl_slice) + 1e-9)
                ax.imshow(tpl_slice, cmap='gray', origin='lower', aspect=aspects[i])
                
                # Foreground: Warped Subject Edges (Red)
                warped_slice = slices_warped[i]
                warped_slice = (warped_slice - np.min(warped_slice)) / (np.max(warped_slice) - np.min(warped_slice) + 1e-9)
                
                dx = scipy.ndimage.sobel(warped_slice, 0)
                dy = scipy.ndimage.sobel(warped_slice, 1)
                mag = np.hypot(dx, dy)
                edges = mag > (np.max(mag) * 0.25)
                
                edge_overlay = np.zeros(edges.shape + (4,))
                edge_overlay[edges, 0] = 1.0 # Red
                edge_overlay[edges, 3] = 1.0 # Alpha
                
                ax.imshow(edge_overlay, origin='lower', aspect=aspects[i])
                ax.set_title(titles[i])
                ax.axis('off')
            
            plt.suptitle(f"Normalization Check: Warped Subject Edges (Red) on Template (Gray)", fontsize=14)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            
        except Exception as e:
            print(f"Norm QC Plot failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_dummy_template(self):
        print("Creating dummy MNI template...")
        os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
        # Create a 91x109x91 block (standard MNI size)
        data = np.zeros((91, 109, 91))
        # Add a "Brain" (Sphere)
        x, y, z = np.ogrid[:91, :109, :91]
        mask = ((x-45)**2 + (y-54)**2 + (z-45)**2) < 40**2
        data[mask] = 100
        # Save
        affine = np.array([[2,0,0,-90],[0,2,0,-126],[0,0,2,-72],[0,0,0,1]]) # Approx MNI 2mm
        img = nib.Nifti1Image(data, affine)
        nib.save(img, self.template_path)

    def _normalize_intensity(self, data, bins=64):
        d_min, d_max = data.min(), data.max()
        if d_max == d_min: return np.zeros_like(data)
        return np.round((data - d_min) / (d_max - d_min) * (bins - 1)).astype(int)

    def _nmi_loss_function(self, params, ref_data, mov_data):
        # ... (Reuse NMI logic, simplified duplicate for standalone file)
        # Apply transform
        aligned = self._apply_transform(mov_data, params, ref_shape=ref_data.shape, order=0)
        
        # Histograms
        bins = 64
        hist_2d, _, _ = np.histogram2d(ref_data.ravel(), aligned.ravel(), bins=bins, range=[[0, bins], [0, bins]])
        
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        px_nz = px[px > 0]
        py_nz = py[py > 0]
        pxy_nz = pxy[pxy > 0]
        
        hx = -np.sum(px_nz * np.log(px_nz))
        hy = -np.sum(py_nz * np.log(py_nz))
        hxy = -np.sum(pxy_nz * np.log(pxy_nz))
        
        return -((hx + hy) / hxy)

    def _apply_transform(self, vol, params, ref_shape=None, order=1):
        # Unpack 12 params
        sx, sy, sz = params[0:3]
        sh_xy, sh_xz, sh_yz = params[3:6]
        rx, ry, rz = params[6:9]
        tx, ty, tz = params[9:12]
        
        # Scale
        S = np.diag([sx, sy, sz, 1])
        
        # Shear (Simplified)
        Sh = np.eye(4)
        Sh[0,1] = sh_xy; Sh[0,2] = sh_xz; Sh[1,2] = sh_yz
        
        # Rotate
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R_mat = Rz @ Ry @ Rx
        R = np.eye(4); R[:3,:3] = R_mat
        
        # Translate
        T = np.eye(4); T[:3, 3] = [tx, ty, tz]
        
        # Total Matrix M = T @ R @ Sh @ S
        M = T @ R @ Sh @ S
        
        # Extract for affine_transform
        # matrix inputs are the inverse usually, or linear part
        # affine_transform takes 'matrix' (3x3) and 'offset' (3)
        # We pass the full matrix to a custom func or separate?
        # scipy.ndimage.affine_transform(input, matrix, offset)
        # mapping: x_in = matrix @ x_out + offset
        # So we need Inverse(M).
        
        try:
            M_inv = np.linalg.inv(M)
        except:
            M_inv = np.eye(4)
            
        mat_3x3 = M_inv[:3, :3]
        offset = M_inv[:3, 3]
        
        output_shape = ref_shape if ref_shape else vol.shape
        return scipy.ndimage.affine_transform(vol, matrix=mat_3x3, offset=offset, output_shape=output_shape, order=order)

    def _params_to_matrix(self, params):
        # Reconstruct M (Forward) for saving
        # ... (Same construction as above)
        sx, sy, sz = params[0:3]
        sh_xy, sh_xz, sh_yz = params[3:6]
        rx, ry, rz = params[6:9]
        tx, ty, tz = params[9:12]
        
        S = np.diag([sx, sy, sz, 1])
        Sh = np.eye(4); Sh[0,1]=sh_xy; Sh[0,2]=sh_xz; Sh[1,2]=sh_yz
        
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R_mat = Rz @ Ry @ Rx
        R = np.eye(4); R[:3,:3] = R_mat
        
        T = np.eye(4); T[:3, 3] = [tx, ty, tz]
        
        M = T @ R @ Sh @ S
        return M
