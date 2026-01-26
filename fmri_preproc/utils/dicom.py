
import os
import shutil
import subprocess
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

class DicomConverter:
    """
    Wrapper for dcm2niix to convert DICOM files to NIfTI format.
    """
    
    def __init__(self, dcm2niix_path: Optional[str] = None):
        """
        Initialize the converter.
        
        Args:
            dcm2niix_path: Path to the dcm2niix executable. If None, tries to find it in PATH.
        """
        self.dcm2niix_path = dcm2niix_path or shutil.which("dcm2niix")
        if not self.dcm2niix_path:
             # Check current directory for dcm2niix.exe (common in Windows portable setups)
             local_exe = os.path.join(os.getcwd(), "dcm2niix.exe")
             if os.path.exists(local_exe):
                 self.dcm2niix_path = local_exe
                 logger.info(f"Using local dcm2niix: {local_exe}")
             else:
                 logger.warning("dcm2niix not found in PATH or local directory.")

    def check_installed(self) -> bool:
        """Check if dcm2niix is available."""
        return self.dcm2niix_path is not None and os.path.exists(self.dcm2niix_path)

    def run(self, input_dir: str, output_dir: str, subject: str, session: Optional[str] = None) -> Tuple[bool, str]:
        """
        Run dcm2niix conversion.
        
        Args:
            input_dir: Directory containing DICOM files.
            output_dir: Directory to save NIfTI files.
            subject: Subject ID (e.g., 'sub-01').
            session: Optional Session ID (e.g., 'ses-01').
            
        Returns:
            Tuple (success: bool, message: str)
        """
        if not self.check_installed():
            logger.error(f"dcm2niix path validation failed. Path: {self.dcm2niix_path}")
            return False, f"dcm2niix is not installed or not found. Path: {self.dcm2niix_path}"

        if not os.path.exists(input_dir):
            return False, f"Input directory does not exist: {input_dir}"

        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting conversion.")
        logger.info(f"Converter: {self.dcm2niix_path}")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")

        # Construct filename pattern: subject_session_...
        # dcm2niix patterns: %p=protocol, %t=time, %s=series number
        filename_pattern = f"{subject}_"
        if session:
            filename_pattern += f"{session}_"
        filename_pattern += "%p_%t_%s"

        cmd = [
            self.dcm2niix_path,
            "-z", "y",             # Gzip output
            "-f", filename_pattern, # Filename pattern
            "-o", output_dir,      # Output directory
            "-b", "y",             # Save BIDS sidecar
            input_dir              # Input directory
        ]
        
        logger.info(f"Running conversion: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Post-process: Organize into anat/func
            self._organize_bids(output_dir)
            
            return True, f"Conversion successful. Output:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            logger.error(f"dcm2niix failed: {e.stderr}")
            return False, f"Conversion failed: {e.stderr}"
        except Exception as e:
            logger.exception("Unexpected error during conversion")
            return False, str(e)

    def _organize_bids(self, output_dir: str):
        """
        Organize flat dcm2niix output into anat/ and func/ folders.
        Also renames files to ensure they have BIDS suffixes (_T1w, _bold)
        so pybids can find them.
        """
        anat_dir = os.path.join(output_dir, 'anat')
        func_dir = os.path.join(output_dir, 'func')
        os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(func_dir, exist_ok=True)
        
        for fname in os.listdir(output_dir):
            full_path = os.path.join(output_dir, fname)
            if not os.path.isfile(full_path):
                continue
                
            name_lower = fname.lower()
            target_dir = None
            new_name = fname
            
            # Identify Anatomical
            if any(x in name_lower for x in ['t1', 'mprage', 'anat']):
                target_dir = anat_dir
                # Ensure _T1w suffix
                base, ext = self._split_ext(fname)
                if not base.endswith('_T1w'):
                    new_name = f"{base}_T1w{ext}"
            
            # Identify Functional
            elif any(x in name_lower for x in ['bold', 'func', 'epi', 'fmri', 'rest', 'task']):
                target_dir = func_dir
                # Ensure _bold suffix
                base, ext = self._split_ext(fname)
                if not base.endswith('_bold'):
                    new_name = f"{base}_bold{ext}"
            
            if target_dir:
                dest_path = os.path.join(target_dir, new_name)
                shutil.move(full_path, dest_path)
                logger.info(f"Moved and renamed {fname} -> {target_dir}/{new_name}")

        # Ensure dataset_description.json exists in the BIDS root (parent of subject dir)
        # Assumes output_dir is .../sub-XX
        parent_dir = os.path.dirname(output_dir)
        desc_path = os.path.join(parent_dir, "dataset_description.json")
        if not os.path.exists(desc_path):
            import json
            try:
                with open(desc_path, 'w') as f:
                    json.dump({
                        "Name": "Converted Dataset",
                        "BIDSVersion": "1.8.0",
                        "DatasetType": "raw"
                    }, f, indent=4)
                logger.info(f"Created BIDS root description at {desc_path}")
            except Exception as e:
                logger.warning(f"Could not create dataset_description.json at {parent_dir}: {e}")

    def _split_ext(self, fname: str) -> Tuple[str, str]:
        """Helper to split filename and extension (handling .nii.gz)"""
        if fname.endswith('.nii.gz'):
            return fname[:-7], '.nii.gz'
        base, ext = os.path.splitext(fname)
        return base, ext
