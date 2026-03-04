
import argparse
import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realign.loader import VolumeLoader
from realign.estimate import MotionEstimator
from realign.transform import TransformBuilder
from realign.reslice import Reslicer

def run_realign(input_file):
    """
    Runs the Realign pipeline on a single input file.
    """
    print(json.dumps({"status": "loading", "message": f"Loading {input_file}..."}))
    

    loader = VolumeLoader(input_file)
    data, affine, header = loader.load()
    outputs = loader.get_output_filenames()
    
    print(json.dumps({"status": "estimating", "message": "Estimating motion..."}))
    

    estimator = MotionEstimator(data, affine)
    motion_params = estimator.estimate_motion()
    
    print(json.dumps({"status": "reslicing", "message": "Reslicing volumes..."}))
    

    matrices = TransformBuilder.build_all_matrices(motion_params)
    

    reslicer = Reslicer(data, affine, header, outputs)
    resliced_data = reslicer.reslice(matrices)
    reslicer.save_outputs(resliced_data, motion_params)
    
    result = {
        "status": "completed",
        "outputs": {k: str(v) for k, v in outputs.items()}
    }
    print(json.dumps(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Realign (Estimate & Reslice)")
    parser.add_argument("input_file", help="Path to input 4D NIfTI file")
    args = parser.parse_args()
    
    try:
        run_realign(args.input_file)
    except Exception as e:
        error = {"status": "error", "message": str(e)}
        print(json.dumps(error))
        sys.exit(1)
