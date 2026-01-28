
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
        # realigned_bold: path to realigned functional (for DVARS)
        # coreg_plot: path to coregistration QC image
        # seg_plot: path to segmentation QC image
        # tissue_maps: list of paths to PVE maps (CSF, GM, WM)
        # norm_plot: path to normalization QC image
        # smooth_plot: path to smoothing QC image
        # qc_metrics: dict of scalar checks (e.g. TR match)

    def execute(self, context: Dict[str, Any]):
        sub = self.inputs['subject']
        out_dir = self.inputs['output_dir']
        # Collect any metadata/metrics passed via context or inputs
        metrics = self.inputs.get('qc_data', {}).copy() # Copy to avoid mutation
        
        print(f"[{self.name}] Generating QC Report for {sub}")
        
        # Ensure directory exists
        import os
        import numpy as np
        import nibabel as nib
        import shutil
        os.makedirs(out_dir, exist_ok=True)
        
        report_path = f"{out_dir}/report.html"
        
        plots = {}
        
        # 1. Motion Analysis
        motion_file = self.inputs.get('motion_params')
        realigned_file = self.inputs.get('realigned_bold')
        if motion_file and os.path.exists(motion_file):
            # Calculate metrics
            mot_metrics = self._calculate_motion_metrics(motion_file, realigned_file)
            metrics.update(mot_metrics)
            
            # Plot
            plot_path = f"{out_dir}/motion_plot.png"
            if self._generate_motion_plot(motion_file, plot_path, mot_metrics):
                plots['Motion Correction'] = "motion_plot.png"
        
        # 2. Coregistration Check
        if self.inputs.get('coreg_plot') and os.path.exists(self.inputs.get('coreg_plot')):
             # Copy to QC dir
             src = self.inputs.get('coreg_plot')
             dst = f"{out_dir}/coreg_check.png"
             shutil.copy(src, dst)
             plots['Coregistration'] = "coreg_check.png"

        # 3. Segmentation Analysis
        tissue_maps = self.inputs.get('tissue_maps')
        if tissue_maps and all(os.path.exists(p) for p in tissue_maps):
             seg_metrics = self._calculate_segmentation_metrics(tissue_maps)
             metrics.update(seg_metrics)

        if self.inputs.get('seg_plot') and os.path.exists(self.inputs.get('seg_plot')):
             src = self.inputs.get('seg_plot')
             dst = f"{out_dir}/seg_check.png"
             shutil.copy(src, dst)
             plots['Segmentation'] = "seg_check.png"

        # 4. Normalization Check
        if self.inputs.get('norm_plot') and os.path.exists(self.inputs.get('norm_plot')):
             src = self.inputs.get('norm_plot')
             dst = f"{out_dir}/norm_check.png"
             shutil.copy(src, dst)
             plots['Normalization'] = "norm_check.png"
             
        # 5. Smoothing Check
        if self.inputs.get('smooth_plot') and os.path.exists(self.inputs.get('smooth_plot')):
             src = self.inputs.get('smooth_plot')
             dst = f"{out_dir}/smooth_check.png"
             shutil.copy(src, dst)
             plots['Smoothing'] = "smooth_check.png"

        QCReport().generate(sub, report_path, metrics, plots)
        
        # Save metrics for ML
        import json
        metrics_path = f"{out_dir}/qc_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        self.outputs['report_file'] = report_path
        self.outputs['metrics_file'] = metrics_path
        print(f"[{self.name}] QC Report saved to {report_path}")
        print(f"[{self.name}] QC Metrics saved to {metrics_path}")

    def _calculate_motion_metrics(self, par_file, bold_file=None):
        import numpy as np
        import nibabel as nib
        stats = {}
        decision = "PASS" # Default
        reasons = []
        
        try:
            # Load params (6 columns: Rx Ry Rz Tx Ty Tz)
            params = np.loadtxt(par_file)
            
            diff = np.zeros_like(params)
            diff[1:] = np.abs(np.diff(params, axis=0))
            
            rot_fd = np.sum(diff[:, 0:3], axis=1) * 50.0 
            trans_fd = np.sum(diff[:, 3:6], axis=1)
            fd = rot_fd + trans_fd
            
            mean_fd = float(np.mean(fd))
            max_fd = float(np.max(fd))
            pct_outliers = float(np.mean(fd > 0.5) * 100.0)
            
            stats['Mean FD (mm)'] = round(mean_fd, 3)
            stats['Max FD (mm)'] = round(max_fd, 3)
            stats['% Volumes FD > 0.5mm'] = round(pct_outliers, 1)
            
            # Decision Logic
            if mean_fd > 0.5:
                decision = "FAIL"
                reasons.append("High Mean FD (> 0.5mm)")
            elif mean_fd > 0.2:
                decision = "WARN" if decision != "FAIL" else "FAIL"
                reasons.append("Moderate Motion (> 0.2mm)")
                
            if pct_outliers > 20.0:
                 decision = "FAIL" if decision != "FAIL" else "FAIL"
                 reasons.append("Too many outliers (> 20%)")
            
            if bold_file and os.path.exists(bold_file):
                 try:
                     img = nib.load(bold_file)
                     data = img.get_fdata() # X,Y,Z,T
                     
                     # DVARS Calculation
                     # Mask background to avoid noise
                     # Simple threshold: > 10% of global mean
                     global_mean = np.mean(data)
                     mask = np.mean(data, axis=3) > (global_mean * 0.1)
                     
                     # Check if mask is empty
                     if np.sum(mask) > 0:
                         masked_data = data[mask].T # (T, V)
                         
                         diff_data = np.diff(masked_data, axis=0) # (T-1, V)
                         # DVARS(t) = sqrt( mean( (I_t - I_{t-1})^2 ) )
                         dvars = np.sqrt(np.mean(diff_data**2, axis=1))
                         
                         # Insert 0 for first frame to match time
                         # stats needs scalar mean
                         mean_dvars = float(np.mean(dvars))
                         stats['Mean DVARS'] = round(mean_dvars, 2)
                     else:
                         stats['Mean DVARS'] = 0.0
                         
                 except Exception as e:
                     print(f"DVARS calc failed: {e}")

            stats['QC Decision'] = decision
            if reasons:
                stats['QC Reasons'] = "; ".join(reasons)
            else:
                stats['QC Reasons'] = "Clean scan"
                 
        except Exception as e:
            print(f"Warning: Failed to calculate motion metrics: {e}")
            
        return stats

    def _calculate_segmentation_metrics(self, tissue_paths):
        import nibabel as nib
        import numpy as np
        stats = {}
        try:
             # Assume order: CSF, GM, WM (pve_0, pve_1, pve_2)
             names = ['CSF', 'GM', 'WM']
             total_vol = 0
             vols = []
             
             for i, path in enumerate(tissue_paths):
                 if i >= 3: break
                 img = nib.load(path) # PVE map (probabilities 0-1)
                 data = img.get_fdata()
                 voxel_vol = np.prod(img.header.get_zooms()[:3]) # mm^3
                 
                 # Volume = Sum(Prob) * Voxel_Size
                 vol_mm3 = np.sum(data) * voxel_vol
                 vols.append(vol_mm3)
                 total_vol += vol_mm3
                 
             for name, vol in zip(names, vols):
                 stats[f'{name} Volume (cm³)'] = round(vol / 1000.0, 2)
                 stats[f'{name} Fraction (%)'] = round((vol / total_vol) * 100.0, 1) if total_vol > 0 else 0
                 
        except Exception as e:
             print(f"Warning: Failed to calculate segmentation metrics: {e}")
        return stats


    def _generate_motion_plot(self, par_file, out_path, metrics=None):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load .par (assuming whitespace separation)
            data = np.loadtxt(par_file)
            # FSL output is usually 6 cols: Rx Ry Rz Tx Ty Tz
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Rotations (first 3 cols) - usually in radians
            ax1.plot(data[:, 0:3])
            ax1.set_ylabel('Rotation (radians)')
            ax1.set_title('Motion Parameters: Rotation')
            ax1.legend(['x', 'y', 'z'], loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Translations (last 3 cols) - usually in mm
            ax2.plot(data[:, 3:6])
            ax2.set_ylabel('Translation (mm)')
            ax2.set_title('Motion Parameters: Translation')
            # ax2.legend(['x', 'y', 'z'], loc='upper right') # maybe too cluttered?
            ax2.legend(['x', 'y', 'z'], loc='upper right')
            ax2.set_xlabel('Volume')
            ax2.grid(True, alpha=0.3)
            
            # Annotate metrics if provided
            if metrics:
                # Text box properties
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # Create text string
                textstr = '\n'.join([
                     f"Mean FD: {metrics.get('Mean FD (mm)', 'N/A'):.2f} mm",
                     f"Max FD: {metrics.get('Max FD (mm)', 'N/A'):.2f} mm",
                     f"% > 0.5mm: {metrics.get('% Volumes FD > 0.5mm', 'N/A'):.1f}%"
                ])
                if 'Mean DVARS' in metrics:
                    textstr += f"\nDVARS: {metrics['Mean DVARS']:.1f}"
                
                # Place text in upper left of top plot
                ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            return True
        except Exception as e:
            print(f"Failed to generate motion plot: {e}")
            return False

