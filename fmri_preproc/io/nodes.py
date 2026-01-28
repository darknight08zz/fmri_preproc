
from typing import Dict, Any, List
import os
from fmri_preproc.core.node import Node
from fmri_preproc.io.bids import BIDSDataset

class DataIngestNode(Node):
    """
    Ingests BIDS data for a specific subject.
    Inputs:
        - bids_dir: Path to BIDS root
        - subject: Subject ID (e.g. 'sub-01')
    Outputs:
        - subject_data: Dict containing 'anat' and 'func' file paths.
    """
    def __init__(self, name: str = "DataIngest"):
        super().__init__(name)
        self.required_inputs = ['bids_dir', 'subject']
        
    def execute(self, context: Dict[str, Any]):
        bids_dir = self.inputs['bids_dir']
        sub = self.inputs['subject']
        
        print(f"[{self.name}] Scanning BIDS dir: {bids_dir} for {sub}")
        
        # Initialize BIDS helper
        ds = BIDSDataset(bids_dir)
        scans = ds.get_scans(sub)
        
        # Validation
        if not scans['anat'] and not scans['func']:
            print(f"[{self.name}] WARNING: No data found for {sub}!")
        
        # We simplify the output structure for downstream nodes
        # Assuming single T1w for now as per pipeline spec
        t1_file = scans['anat'][0]['path'] if scans['anat'] else None
        
        # Functional files
        func_files = [f['path'] for f in scans['func']]
        
        output_data = {
            'subject': sub,
            'anat': t1_file,
            'func': func_files,
            'meta': {} # Place to store parsed metadata if needed
        }
        
        self.outputs['subject_data'] = output_data
        
        # Also set individual outputs for convenience if nodes want simple strings
        self.outputs['anat_file'] = t1_file
        self.outputs['func_files'] = func_files
        # Temporary: Output single string for first run to satisfy simple DAG nodes
        self.outputs['func_file'] = func_files[0] if func_files else None
        
        print(f"[{self.name}] Found 1 T1w and {len(func_files)} BOLD series.")
