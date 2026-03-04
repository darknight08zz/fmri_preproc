import sys
import os
import argparse
from dicom2nifti_spm.converter import convert_directory

def main():
    parser = argparse.ArgumentParser(description="Native DICOM to NIfTI Converter (SPM-style)")
    parser.add_argument("input_dir", help="Directory containing DICOM files")
    parser.add_argument("output_dir", help="Directory to save NIfTI files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        sys.exit(1)
        
    convert_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
