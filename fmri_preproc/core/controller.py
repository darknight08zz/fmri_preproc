
from typing import Dict, List, Optional
import os
from fmri_preproc.core.workflow import Workflow
from fmri_preproc.io.nodes import DataIngestNode
from fmri_preproc.io.bids import BIDSDataset
from fmri_preproc.func.nodes import (
    DummyScanNode, RealignmentNode, SliceTimingNode, 
    CoregistrationNode, NormalizationNode, SmoothingNode
)
from fmri_preproc.anat.nodes import SegmentationNode
from fmri_preproc.qc.nodes import QCNode

class PipelineController:
    """
    Orchestrates the fMRI Preprocessing DAG.
    """
    def __init__(self, output_root: str):
        self.output_root = output_root

    def build_workflow(self, subject: str, bids_dir: str, has_anat: bool) -> Workflow:
        wf = Workflow(f"Workflow_{subject}")
        
        # 0. Setup Paths
        subj_out = os.path.join(self.output_root, subject)
        
        # 1. Ingest
        ingest = DataIngestNode("Ingest")
        ingest.set_input('bids_dir', bids_dir)
        ingest.set_input('subject', subject)
        wf.add_node(ingest)

        # 2. Dummy Scans
        dummy = DummyScanNode("Dummy")
        wf.connect(ingest, 'func_file', dummy, 'bold_file') 
        
        # 3. Realignment
        realign = RealignmentNode("Realign")
        wf.connect(dummy, 'trimmed_bold', realign, 'bold_file')
        
        # 4. Slice Timing
        stc = SliceTimingNode("STC")
        stc.set_input('tr', 2.0) # Param should be configurable or read from BIDS
        wf.connect(realign, 'realigned_bold', stc, 'bold_file')
        
        last_func_node = stc
        last_func_out = 'stc_bold'
        
        if has_anat:
            print(f"[Controller] Anatomical data found. Building FULL pipeline (Coreg -> Seg -> Norm).")
            # 5. Coregistration (MeanFunc -> Anat)
            coreg = CoregistrationNode("Coreg")
            wf.connect(realign, 'mean_bold', coreg, 'ref_func')
            wf.connect(ingest, 'anat_file', coreg, 'source_anat')
            
            # 6. Segmentation (Coreg Anat -> Deformation)
            seg = SegmentationNode("Segmentation")
            wf.connect(coreg, 'coreg_anat', seg, 'anat_file')
            
            # 7. Normalization
            norm = NormalizationNode("Normalization")
            wf.connect(stc, 'stc_bold', norm, 'bold_file')
            wf.connect(seg, 'deformation_field', norm, 'deformation_field')
            
            last_func_node = norm
            last_func_out = 'normalized_bold'
        else:
            print(f"[Controller] WARNING: No anatomical data. Building FUNCTIONAL-ONLY pipeline (Skipping Coreg, Seg, Norm).")
            # Fallback: Just skip to smoothing in native space
        
        # 8. Smoothing
        smooth = SmoothingNode("Smoothing")
        smooth.set_input('fwhm', [6, 6, 6])
        wf.connect(last_func_node, last_func_out, smooth, 'bold_file')
        
        # 9. QC
        qc = QCNode("QC")
        qc.set_input('subject', subject)
        qc.set_input('output_dir', os.path.join(subj_out, 'qc'))
        
        # Connect available QC metrics
        wf.connect(realign, 'motion_params', qc, 'motion_params')
        wf.connect(stc, 'stc_bold', qc, 'func_file') # Just for reference if needed
        
        if has_anat:
            wf.connect(coreg, 'qc_plot', qc, 'coreg_plot')
            wf.connect(seg, 'seg_qc_plot', qc, 'seg_plot')
            wf.connect(seg, 'warp_qc_plot', qc, 'norm_plot') # "Normalization" check
        
        wf.connect(smooth, 'qc_plot', qc, 'smooth_plot')
        

        
        return wf

    def run(self, subject: str, bids_dir: str, status_callback=None):
        print(f"Initializing pipeline for {subject}...")
        
        # Check for Anatomical Data Availability
        has_anat = False
        try:
            ds = BIDSDataset(bids_dir)
            # Pre-scan checks to decide topology
            # Note: We need to handle the strict ID matching that we fixed in DataIngest too?
            # Actually DataIngest uses BIDSDataset internally.
            # We can rely on BIDSDataset logic.
            
            # BIDS wrapper handles stripping 'sub-' now? 
            # Yes, I updated bids.py to handle it.
            scans = ds.get_scans(subject)
            if scans['anat']:
                has_anat = True
        except Exception as e:
            print(f"Error checking BIDS content: {e}")
        
        wf = self.build_workflow(subject, bids_dir, has_anat)
        wf.run(status_callback=status_callback)
