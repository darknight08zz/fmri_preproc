
from typing import Dict, Any, List
import os
from fmri_preproc.core.node import Node
from fmri_preproc.anat.segmentation import Segmentation
from fmri_preproc.anat.normalization import Normalization as WarpEstimation

class SegmentationNode(Node):
    """
    Node for Anatomical Segmentation & Warp Estimation (Unified Segment-like).
    Inputs:
        - anat_file (str) (Coregistered T1w)
        - template_path (str) (Optional, for spatial normalization)
    Outputs:
        - tissue_maps (List[str]) (pve_0, pve_1, etc.)
        - bias_corrected (str) (m_prefix)
        - deformation_field (str) (y_prefix)
        - forward_transforms (List[str]) (Affine + Warp)
        - seg_qc_plot (str)
        - warp_qc_plot (str)
    """
    def __init__(self, name: str = "Segmentation"):
        super().__init__(name)
        self.required_inputs = ['anat_file']

    def execute(self, context: Dict[str, Any]):
        t1_in = self.inputs['anat_file']
        template = self.inputs.get('template_path', "templates/MNI152_T1_2mm.nii.gz")
        
        dirname, filename = os.path.split(t1_in)
        
        # 1. Run FAST (Tissue Segmentation + Bias Field)
        # SPM "m" prefix for bias corrected.
        # FAST doesn't easily output "m" prefix automatically, usually outputs restored image.
        # We'll use our wrapper's convention.
        
        # Output base
        file_base = os.path.splitext(os.path.splitext(filename)[0])[0] # handle .nii.gz
        seg_base = os.path.join(dirname, f"{file_base}")
        
        print(f"[{self.name}] Running Segmentation (FAST) on {t1_in}")
        # Note: Segmentation.run copies to pve files etc.
        # Ideally our wrapper should be more explicit about returning path to bias-corrected.
        # Currently defaults to just returning True.
        # We assume standard output naming from FSL FAST.
        Segmentation().run(t1_in, seg_base)
        # Assuming wrapper returns (bool, path)
        success, seg_qc_plot = Segmentation().run(t1_in, seg_base)
        
        # Gather outputs
        tissue_maps = [f"{seg_base}_pve_{i}.nii.gz" for i in range(3)]
        bias_corrected = f"{seg_base}_restore.nii.gz" # FAST default for bias corrected
        
        # 2. Run Warp Estimation (Normalization logic)
        # In SPM, Segmentation produces the deformation field 'y_'.
        # We use ANTs to estimate T1 -> Template.
        
        warp_out_base = os.path.join(dirname, f"y_{file_base}") # y_ prefix
        print(f"[{self.name}] Estimating Deformations (ANTs) -> {warp_out_base}")
        
        warped_img, transforms, warp_qc_plot = WarpEstimation(template_path=template).run(t1_in, warp_out_base)
        
        # Identify the Warp Field (nonlinear)
        # transforms list usually [Warp, Affine] or similar.
        # We export the list for full usage, and specifically the warp if needed.
        
        self.outputs['tissue_maps'] = tissue_maps
        self.outputs['bias_corrected'] = bias_corrected
        self.outputs['deformation_field'] = transforms[0] if transforms else None # Heuristic
        self.outputs['forward_transforms'] = transforms
        self.outputs['seg_qc_plot'] = seg_qc_plot
        self.outputs['warp_qc_plot'] = warp_qc_plot
        
        print(f"[{self.name}] Finished Segmentation & Warp Estimation.")
