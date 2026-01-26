import os
import shutil
import glob
from typing import List
import nibabel as nib
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.fsl import Merge

class SpatialTransforms:
    """
    Applies multiple spatial transforms using Nipype (ANTs ApplyTransforms).
    Supports single-shot resampling for 4D files with per-volume motion matrices.
    """
    def run(self, input_path: str, reference_path: str, output_path: str, transforms: List[str]) -> bool:
        """
        Applies transforms. 
        If 'transforms' contains a directory, it treats it as a folder of per-volume matrices (sorted alphamerically).
        """
        print(f"Running ANTs ApplyTransforms on {input_path}")
        
        # Check for per-volume transform directories
        dynamic_transforms = []
        static_transforms = []
        for t in transforms:
            if os.path.isdir(t):
                dynamic_transforms.append(t)
            else:
                static_transforms.append(t)
        
        if len(dynamic_transforms) > 1:
            raise ValueError("Multiple dynamic transform directories not supported yet.")
            
        try:
            if not dynamic_transforms:
                # Simple case: All static transforms (e.g. just MNI) or applying to 3D image
                self._run_single(input_path, reference_path, output_path, static_transforms)
            else:
                # Complex case: 4D file with one matrix per volume (Motion Correction case)
                self._run_4d_split(input_path, reference_path, output_path, static_transforms, dynamic_transforms[0])
                
            return True
        except Exception as e:
            print(f"SpatialTransforms failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _run_single(self, inp, ref, out, xforms):
        at = ApplyTransforms()
        at.inputs.input_image = inp
        at.inputs.reference_image = ref
        at.inputs.output_image = out
        at.inputs.transforms = xforms
        at.inputs.interpolation = 'LanczosWindowedSinc' # High quality interpolation
        at.inputs.dimension = 3
        # ANTs handles 4D inputs with 3D reference by processing each volume if dimension=3 is set, 
        # but only if transforms are static.
        at.run()

    def _run_4d_split(self, inp, ref, out, static_xforms, mat_dir):
        print(f"Applying per-volume transforms from {mat_dir}")
        img = nib.load(inp)
        if len(img.shape) != 4:
            raise ValueError(f"Expected 4D input for dynamic transforms, got shape {img.shape}")
        
        n_vols = img.shape[3]
        mats = sorted(glob.glob(os.path.join(mat_dir, "*")))
        if len(mats) != n_vols:
            # Fallback: if mock data, we might have mismatch. 
            # If standard MCFLIRT, we expect mismatch if we dropped volumes not updated or something.
            print(f"Warning: Vol count {n_vols} != Mat count {len(mats)}. Truncating/Padding logic might be needed.")
            # Strict for now
            if len(mats) < n_vols:
                 raise ValueError(f"Not enough matrices ({len(mats)}) for volumes ({n_vols})")
            mats = mats[:n_vols] # Trim if too many (e.g. if we have dummy scans handled differently)

        # Temp dir for split processing
        tmp_dir = os.path.join(os.path.dirname(out), "tmp_split_xform")
        os.makedirs(tmp_dir, exist_ok=True)
        
        processed_vols = []
        
        # Split - Apply - Collect
        # We can avoid writing split files by using nilearn/nibabel to save individual vols, 
        # but looping is easiest to debug.
        
        imgs = nib.four_to_three(img)
        
        for i, one_vol_img in enumerate(imgs):
            vol_tmp = os.path.join(tmp_dir, f"vol_{i:04d}.nii.gz")
            out_tmp = os.path.join(tmp_dir, f"out_{i:04d}.nii.gz")
            nib.save(one_vol_img, vol_tmp)
            
            # Combine transforms. Order: [Static (e.g. MNI, Coreg), Dynamic (Motion)]
            # ANTs applies transforms: T1(x) = T2(T1(x)). 
            # Usually: Reference <- [Affine T1->MNI] <- [Affine EPI->T1] <- [Affine Motion EPI(t)->EPI(ref)] <- Source
            # So order in list is [T1->MNI, EPI->T1, MotionMat]
            
            # Note: Nipype ApplyTransforms expects transforms in REVERSE order of application? 
            # No, ANTs convention: -t TransA -t TransB means TransA(TransB(x)).
            # So we want [MNI_to_T1 (if warping ref), T1_to_EPI... wait]
            # Standard: if mapping Source -> Reference.
            # Transforms should be: [Reference -> ... -> Source]
            # So: [T1_to_MNI (Warp), EPI_to_T1 (Affine), Motion (Affine)]
            
            current_xforms = static_xforms + [mats[i]]
            
            self._run_single(vol_tmp, ref, out_tmp, current_xforms)
            processed_vols.append(out_tmp)
            
        # Merge
        merger = Merge()
        merger.inputs.in_files = processed_vols
        merger.inputs.dimension = 't'
        merger.inputs.merged_file = out
        merger.run()
        
        # Cleanup
        shutil.rmtree(tmp_dir)

