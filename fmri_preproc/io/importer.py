import os
import shutil
import json
import nibabel as nib

class BIDSImporter:
    """
    Helper to convert raw NIfTI files into a BIDS-compliant dataset.
    """
    def __init__(self, output_dir: str = "imported_bids_dataset"):
        self.output_dir = output_dir

    def import_subject(self, subject_id: str, t1_path: str, bold_path: str, tr: float):
        """
        Imports a single subject.
        """
        sub = f"sub-{subject_id}"
        sub_dir = os.path.join(self.output_dir, sub)
        anat_dir = os.path.join(sub_dir, "anat")
        func_dir = os.path.join(sub_dir, "func")
        
        os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(func_dir, exist_ok=True)
        
        # 1. Import T1w
        if t1_path and os.path.exists(t1_path):
            ext = ".nii.gz" if t1_path.endswith(".nii.gz") else ".nii"
            dest_t1 = os.path.join(anat_dir, f"{sub}_T1w{ext}")
            shutil.copy(t1_path, dest_t1)
            
            # Create minimal sidecar
            with open(dest_t1.replace(ext, ".json"), "w") as f:
                json.dump({"Modality": "MR", "SeriesDescription": "T1w"}, f, indent=2)
                
        # 2. Import BOLD
        if bold_path and os.path.exists(bold_path):
            ext = ".nii.gz" if bold_path.endswith(".nii.gz") else ".nii"
            dest_bold = os.path.join(func_dir, f"{sub}_task-rest_bold{ext}")
            shutil.copy(bold_path, dest_bold)
            
            # Create critical metadata sidecar
            meta = {
                "RepetitionTime": float(tr),
                "TaskName": "rest",
                "PhaseEncodingDirection": "j", # Default assumption, user can edit
                "SliceTiming": [] # Empty, will trigger warning but allow run
            }
            with open(dest_bold.replace(ext, ".json"), "w") as f:
                json.dump(meta, f, indent=2)
                
        # 3. Create dataset_description.json if not exists
        desc_path = os.path.join(self.output_dir, "dataset_description.json")
        if not os.path.exists(desc_path):
            with open(desc_path, "w") as f:
                json.dump({
                    "Name": "Imported Dataset",
                    "BIDSVersion": "1.8.0",
                    "Authors": ["fMRI Preproc Tool"]
                }, f, indent=2)
                
        return sub_dir
