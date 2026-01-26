import os
import nibabel as nib
import numpy as np
from fmri_preproc.io.bids import BIDSDataset
# Stage 1
from fmri_preproc.anat.bias_correction import BiasCorrection
from fmri_preproc.anat.skull_strip import SkullStripper
from fmri_preproc.anat.segmentation import Segmentation
from fmri_preproc.anat.normalization import Normalization
# Stage 2
from fmri_preproc.func.dummy_scans import DummyScanRemoval
from fmri_preproc.func.slice_timing import SliceTiming
from fmri_preproc.func.motion_correction import MotionCorrection
from fmri_preproc.func.sdc import DistortionCorrection
from fmri_preproc.func.coregistration import Coregistration
from fmri_preproc.func.spatial_transforms import SpatialTransforms
# Stage 3
from fmri_preproc.confounds.motion import MotionMetrics
from fmri_preproc.confounds.extraction import SignalExtraction
from fmri_preproc.confounds.acompcor import ACompCor
# Stage 4/5
from fmri_preproc.postproc.filtering import TemporalFiltering
from fmri_preproc.postproc.scaling import Scaling
from fmri_preproc.postproc.smoothing import SpatialSmoothing
from fmri_preproc.qc.report import QCReport

class PipelineManager:
    def __init__(self, output_root: str):
        self.output_root = output_root

    def _compute_mean(self, in_file, out_file):
        """Helper to compute mean image using nibabel"""
        print(f"Computing mean of {in_file}")
        img = nib.load(in_file)
        data = img.get_fdata()
        # Compute mean across time (last axis)
        if len(data.shape) == 4:
            mean_data = np.mean(data, axis=3)
        else:
            mean_data = data
        
        mean_img = nib.Nifti1Image(mean_data, img.affine, img.header)
        nib.save(mean_img, out_file)
        return out_file

    def run_subject(self, bids_path: str, subject: str):
        print(f"=== Processing Subject {subject} (Single-Shot Resampling Mode) ===")
        
        # 0. Setup directories
        subj_out = os.path.join(self.output_root, subject)
        anat_out = os.path.join(subj_out, "anat")
        func_out = os.path.join(subj_out, "func")
        qc_out = os.path.join(subj_out, "qc")
        
        os.makedirs(anat_out, exist_ok=True)
        os.makedirs(func_out, exist_ok=True)
        os.makedirs(qc_out, exist_ok=True)
        
        ds = BIDSDataset(bids_path)
        scans = ds.get_scans(subject)
        
        t1_raw = None
        if scans['anat']:
             t1_raw = scans['anat'][0]['path']
        
        func_raws = [f['path'] for f in scans['func']]
        
        if not func_raws:
            print("Missing Func coverage. Aborting.")
            return False
            
        # Define T1 artifacts even if None, to keep scope safe
        t1_brain = None
        t1_to_mni_xforms = []
        wm_seg_path = None

        if t1_raw:
            # --- Stage 1: Anat ---
            t1_bias = os.path.join(anat_out, f"{subject}_T1w_bias.nii.gz")
            t1_brain = os.path.join(anat_out, f"{subject}_T1w_brain.nii.gz")
            t1_seg_base = os.path.join(anat_out, f"{subject}_T1w")
            t1_mni = os.path.join(anat_out, f"{subject}_T1w_space-MNI.nii.gz")
            
            BiasCorrection().run(t1_raw, t1_bias)
            SkullStripper().run(t1_bias, t1_brain)
            Segmentation().run(t1_brain, t1_seg_base)
            wm_seg_path = f"{t1_seg_base}_pve_2.nii.gz"
            
            # NEW: Normalization returns transforms
            _, t1_to_mni_xforms = Normalization().run(t1_brain, os.path.join(anat_out, f"{subject}_T1w_space-MNI"))
        else:
            print("No T1w found. Skipping Stage 1 (Anat) and Normalization.")
        
        # --- Stage 2: Func ---
        for func_raw in func_raws:
            basename = os.path.basename(func_raw).replace(".nii.gz", "").replace(".nii", "")
            
            # 1. Dummy Scans
            func_dro = os.path.join(func_out, f"{basename}_desc-dummy.nii.gz")
            DummyScanRemoval().run(func_raw, func_dro)
            
            # 2. Slice Timing
            func_stc = os.path.join(func_out, f"{basename}_desc-stc.nii.gz")
            SliceTiming().run(func_dro, func_stc, tr=2.0)
            
            # 3. Motion Correction (Estimate Only)
            func_moco_qc = os.path.join(func_out, f"{basename}_desc-moco_qc.nii.gz") # Used for QC, not pipeline flow
            # NEW: Returns mats
            _, moco_mats_dir = MotionCorrection().run(func_stc, func_moco_qc)
            
            # Calculate Mean of MoCo for Registration
            func_mean = os.path.join(func_out, f"{basename}_desc-mean.nii.gz")
            self._compute_mean(func_moco_qc, func_mean)
            
            # 4. SDC (Estimate on Mean)
            func_sdc_qc = os.path.join(func_out, f"{basename}_desc-sdc_qc.nii.gz")
            # NEW: Returns warp
            # Assuming we can run SDC on the Mean image to get the warp
            _, sdc_warp = DistortionCorrection().run(func_mean, func_sdc_qc, fmap_path=None) 
            
            # 5. Coregistration (EPI Mean -> T1 Brain)
            func_coreg_qc = os.path.join(func_out, f"{basename}_desc-coreg_qc.nii.gz") # QC only
            epi_to_t1_mat = None
            
            if t1_brain:
                input_for_coreg = func_sdc_qc if sdc_warp else func_mean
                _, epi_to_t1_mat = Coregistration().run(input_for_coreg, t1_brain, func_coreg_qc, wm_seg_path=wm_seg_path)
            
            # 6. Single-Shot Resampling (Transforms)
            # Chain: T1->MNI (List), EPI->T1 (Mat), SDC (Warp), Motion (Dir)
            
            final_transforms = []
            if t1_to_mni_xforms:
                final_transforms.extend(t1_to_mni_xforms)
            if epi_to_t1_mat:
                final_transforms.append(epi_to_t1_mat)
            if sdc_warp:
                final_transforms.append(sdc_warp)
            
            # Always have motion
            final_transforms.append(moco_mats_dir)
            
            func_mni = os.path.join(func_out, f"{basename}_space-MNI.nii.gz")
            print(f"Applying combined transforms: {final_transforms}")
            
            # Apply to STC data (Pre-Motion)
            # If we have NO T1->MNI, this is essentially Native Space resampling (Motion + SDC)
            # which is still valid, just not MNI space.
            ref_img = t1_mni if (t1_raw and t1_mni and os.path.exists(t1_mni)) else func_mean
            if not t1_raw:
                # Rename output to native if we aren't truly moving to MNI
                func_mni = os.path.join(func_out, f"{basename}_desc-preproc.nii.gz")
                
            SpatialTransforms().run(func_stc, ref_img, func_mni, transforms=final_transforms)
            
            # --- Confounds & Postproc ---
            # 7. Confounds (Use MCFLIRT parameters from QC step)
            motion_parp = func_moco_qc.replace(".nii.gz", ".par")
            if os.path.exists(motion_parp):
                 MotionMetrics().calculate_fd(motion_parp)
            
            # 8. Postproc (Filtering/Scaling on MNI data)
            # 8a. Spatial Smoothing
            func_smooth = os.path.join(func_out, f"{basename}_space-MNI_desc-smooth.nii.gz")
            SpatialSmoothing().run(func_mni, func_smooth, fwhm=6.0)

            # 8b. Temporal Filtering
            func_filtered = os.path.join(func_out, f"{basename}_desc-filtered.nii.gz")
            TemporalFiltering().run(func_smooth, func_filtered, highpass=0.01)

            # 8c. Scaling
            func_clean = os.path.join(func_out, f"{basename}_desc-clean.nii.gz")
            Scaling().run(func_filtered, func_clean)
            
        # --- QC ---
        QCReport().generate(subject, os.path.join(qc_out, "report.html"), {"Status": "Success (Single-Shot)"}, {})
        
        print(f"Subject {subject} finished.")
        return True

