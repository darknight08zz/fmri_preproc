
import os
import numpy as np
import nibabel as nib
from pathlib import Path

# Import the modules we just created
from realign.loader import VolumeLoader
from realign.estimate import MotionEstimator
from realign.transform import TransformBuilder
from realign.reslice import Reslicer

def test_pipeline():
    print("=== Testing Realign Pipeline ===")
    
    # 1. Create Synthetic Data (4D)
    # 64x64x32 volume, 10 time points
    dims = (64, 64, 32, 10)
    data = np.zeros(dims)
    
    # Create a "brain" block in the center
    # Add random noise
    center = np.array(dims[:3]) // 2
    r = 15
    y, x, z = np.ogrid[:dims[0], :dims[1], :dims[2]]
    mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= r**2
    
    for t in range(dims[3]):
        vol = np.zeros(dims[:3])
        vol[mask] = 100
        # Add some random motion by shifting the block slightly
        # For simplicity, just translate by t*0.5 mm in x
        # This simulates a linear drift
        shift = int(t * 0.5)
        # Shift data simply by rolling (integer shift)
        if shift > 0:
            vol = np.roll(vol, shift, axis=0) # Shift in X
            
        data[..., t] = vol + np.random.normal(0, 5, dims[:3]) # Add noise

    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    
    # Save test file
    test_file = Path("test_4d.nii")
    nib.save(img, test_file)
    print(f"Created synthetic data: {test_file}")
    
    try:
        # 2. Test Loader
        print("\n--- Testing Loader ---")
        loader = VolumeLoader(test_file)
        data, affine, header = loader.load()
        outputs = loader.get_output_filenames()
        print(f"Loaded data shape: {data.shape}")
        
        # 3. Test Estimator
        print("\n--- Testing Estimator ---")
        estimator = MotionEstimator(data, affine)
        motion_params = estimator.estimate_motion() # Estimate motion relative to first vol
        print(f"Estimated motion parameters (first 5):\n{motion_params[:5]}")
        
        # 4. Test Transform Builder
        print("\n--- Testing Transform Builder ---")
        matrices = TransformBuilder.build_all_matrices(motion_params)
        print(f"Built {matrices.shape[0]} matrices")
        
        # 5. Test Reslicer
        print("\n--- Testing Reslicer ---")
        reslicer = Reslicer(data, affine, header, outputs)
        resliced_data = reslicer.reslice(matrices)
        
        # Save Outputs
        reslicer.save_outputs(resliced_data, motion_params)
        
        # Verify output files exist
        assert outputs['resliced'].exists()
        assert outputs['mean'].exists()
        assert outputs['motion_params'].exists()
        print("\n✅ Pipeline completed successfully! Output files verified.")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup (optional, keeping for inspection)
        # os.remove(test_file)
        pass

if __name__ == "__main__":
    test_pipeline()
