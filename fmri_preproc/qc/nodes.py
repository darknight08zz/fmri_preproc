
from typing import Dict, Any, List
from fmri_preproc.core.node import Node
from fmri_preproc.qc.report import QCReport

class QCNode(Node):
    """
    Node for generating Quality Control reports.
    Inputs:
        - subject (str)
        - qc_data (Dict) (Optional: metrics collected during processing)
        - output_dir (str)
    Outputs:
        - report_file (str)
    """
    def __init__(self, name: str = "QC"):
        super().__init__(name)
        self.required_inputs = ['subject', 'output_dir']
        # Optional inputs for plotting
        # motion_params: path to .par file
        # coreg_plot: path to coregistration QC image
        # seg_plot: path to segmentation QC image
        # norm_plot: path to normalization QC image
        # smooth_plot: path to smoothing QC image
        # qc_metrics: dict of scalar checks (e.g. TR match)

        
    def execute(self, context: Dict[str, Any]):
        sub = self.inputs['subject']
        out_dir = self.inputs['output_dir']
        # Collect any metadata/metrics passed via context or inputs
        metrics = self.inputs.get('qc_data', {})
        
        print(f"[{self.name}] Generating QC Report for {sub}")
        
        # Ensure directory exists
        import os
        os.makedirs(out_dir, exist_ok=True)
        
        report_path = f"{out_dir}/report.html"
        
        # Using existing QCReport class
        # generate(subject, out_file, status_dict, image_paths)
        # generate(subject, out_file, status_dict, image_paths)
        plots = {}
        
        # 1. Motion Plot
        motion_file = self.inputs.get('motion_params')
        if motion_file and os.path.exists(motion_file):
            plot_path = f"{out_dir}/motion_plot.png"
            if self._generate_motion_plot(motion_file, plot_path):
                plots['Motion Correction'] = "motion_plot.png"
        
        # 2. Coregistration Check
        if self.inputs.get('coreg_plot') and os.path.exists(self.inputs.get('coreg_plot')):
             # Copy to QC dir
             import shutil
             src = self.inputs.get('coreg_plot')
             dst = f"{out_dir}/coreg_check.png"
             shutil.copy(src, dst)
             plots['Coregistration'] = "coreg_check.png"

        # 3. Segmentation Check
        if self.inputs.get('seg_plot') and os.path.exists(self.inputs.get('seg_plot')):
             # Copy
             src = self.inputs.get('seg_plot')
             dst = f"{out_dir}/seg_check.png"
             shutil.copy(src, dst)
             plots['Segmentation'] = "seg_check.png"

        # 4. Normalization Check
        if self.inputs.get('norm_plot') and os.path.exists(self.inputs.get('norm_plot')):
             # Copy
             src = self.inputs.get('norm_plot')
             dst = f"{out_dir}/norm_check.png"
             shutil.copy(src, dst)
             plots['Normalization'] = "norm_check.png"
             
        # 5. Smoothing Check
        if self.inputs.get('smooth_plot') and os.path.exists(self.inputs.get('smooth_plot')):
             # Copy
             src = self.inputs.get('smooth_plot')
             dst = f"{out_dir}/smooth_check.png"
             shutil.copy(src, dst)
             plots['Smoothing'] = "smooth_check.png"

        QCReport().generate(sub, report_path, metrics, plots)
        
        self.outputs['report_file'] = report_path
        self.outputs['report_file'] = report_path
        print(f"[{self.name}] QC Report saved to {report_path}")

    def _generate_motion_plot(self, par_file, out_path):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load .par (assuming whitespace separation)
            data = np.loadtxt(par_file)
            # FSL output is usually 6 cols: Rx Ry Rz Tx Ty Tz
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            # Rotations (first 3 cols) - usually in radians
            ax1.plot(data[:, 0:3])
            ax1.set_ylabel('Rotation (radians)')
            ax1.set_title('Motion Parameters: Rotation')
            ax1.legend(['x', 'y', 'z'], loc='upper right')
            
            # Translations (last 3 cols) - usually in mm
            ax2.plot(data[:, 3:6])
            ax2.set_ylabel('Translation (mm)')
            ax2.set_title('Motion Parameters: Translation')
            ax2.legend(['x', 'y', 'z'], loc='upper right')
            ax2.set_xlabel('Volume')
            
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            return True
        except Exception as e:
            print(f"Failed to generate motion plot: {e}")
            return False

