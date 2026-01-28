
import os
import sys
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fmri_preproc.func.slice_timing import SliceTiming

def test_native_stc():
    output_dir = "tests/test_output_stc"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 1. Create Synthetic Data
    # 5 Slices, 20 Volumes, TR=2.0s
    n_slices = 5
    n_vols = 20
    tr = 2.0
    x_dim, y_dim = 10, 10
    
    data = np.zeros((x_dim, y_dim, n_slices, n_vols))
    
    # Generate a sine wave signal: sin(2*pi*f*t)
    # Frequency f = 0.1 Hz (Period 10s => 5 volumes)
    freq = 0.1
    
    # We will pretend the signal is identical in all slices in the "Real World"
    # But because slices are acquired at different times, the recorded signal in the voxel
    # will be phase shifted if we just stack them as "Volume k".
    # Wait, usually we simulate:
    # True signal S(t).
    # Slice z acquired at t_z = k*TR + offset_z.
    # Recorded Data[k, z] = S(k*TR + offset_z).
    
    # Let's construct Data that matches this.
    # Offsets: 0, 0.4, 0.8, 1.2, 1.6 (Ascending)
    offsets = [z * (tr / n_slices) for z in range(n_slices)]
    
    times_vol = np.arange(n_vols) * tr
    
    for z in range(n_slices):
        # Actual acquisition times for this slice
        t_acq = times_vol + offsets[z]
        # Signal
        sig = np.sin(2 * np.pi * freq * t_acq)
        data[:, :, z, :] = sig
        
    in_file = os.path.join(output_dir, "test_bold.nii.gz")
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), in_file)
    
    # 2. Run STC
    # Target Reference: TR/2 = 1.0s
    out_file = os.path.join(output_dir, "stc_bold.nii.gz")
    
    print("\n[Test] Running Slice Timing...")
    success = SliceTiming().run(in_file, out_file, tr, slice_timing=offsets)
    
    if not success:
        print("FAILED: SliceTiming.run returned False")
        sys.exit(1)
        
    # 3. Verify
    # The output should resample signal to t_ref = k*TR + 1.0
    # Expected Output[k, z] = S(k*TR + 1.0)
    # Since S(t) is global sine wave, Output[k, z] should be identical for all z (ignoring interpolation error)
    # because we aligned them all to the same reference time!
    
    out_img = nib.load(out_file)
    out_data = out_img.get_fdata()
    
    # Check alignment across slices
    # Pick a voxel
    x, y = 5, 5
    
    # Plotting for visual confirmation if run locally
    # We'll just assert error is small
    
    # Expected signal at Reference Time
    ref_times = times_vol + (tr / 2.0)
    expected_sig = np.sin(2 * np.pi * freq * ref_times)
    
    errors = []
    print("\n[Test] Checking Error per Slice (Slice 0 should move most, Mid slice least if Ref=1.0):")
    # Actually if offsets are 0..1.6. Mid is 0.8. Ref is 1.0.
    
    for z in range(n_slices):
        corrected_sig = out_data[x, y, z, :]
        
        # RMSE
        # Ignore edges where extrapolation might be imperfect?
        mse = np.mean((corrected_sig[2:-2] - expected_sig[2:-2])**2)
        rmse = np.sqrt(mse)
        errors.append(rmse)
        print(f"  Slice {z} (Offset {offsets[z]:.2f}s) -> RMSE: {rmse:.5f}")
        
    mean_rmse = np.mean(errors)
    print(f"\n[Test] Mean RMSE: {mean_rmse:.5f}")
    
    if mean_rmse < 0.1: # Sine amplitude 1.0. 10% error is generous for cubic spline on coarse grid
        print("SUCCESS: Signal aligned successfully.")
    else:
        print("FAILED: Error too high.")
        sys.exit(1)

if __name__ == "__main__":
    test_native_stc()
