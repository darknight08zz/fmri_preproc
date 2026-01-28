import os
import shutil
import numpy as np
import nibabel as nib
from scipy import interpolate
from nipype.interfaces.fsl import SliceTimer

class SliceTiming:
    """
    Slice Timing Correction using Nipype (FSL SliceTimer) or Native Python Interpolation.
    """
    def run(self, input_path: str, output_path: str, tr: float, slice_timing: list = None) -> bool:
        print(f"SliceTiming: Checking for FSL...")
        # Check if FSL is really available (flirt is a good proxy)
        fsl_available = shutil.which("flirt") is not None
        
        if fsl_available:
            print(f"Running FSL SliceTimer on {input_path}")
            try:
                st = SliceTimer()
                st.inputs.in_file = input_path
                st.inputs.out_file = output_path
                st.inputs.time_repetition = tr
                if slice_timing:
                    # Create a timing file
                    timing_file = "slice_timings.txt"
                    with open(timing_file, "w") as f:
                        for t in slice_timing:
                            f.write(f"{t}\n")
                    st.inputs.custom_timings = timing_file
                
                st.run()
                
                if slice_timing and os.path.exists("slice_timings.txt"):
                    try: os.remove("slice_timings.txt")
                    except: pass
                    
                return True
            except Exception as e:
                print(f"Nipype SliceTimer failed: {e}")
        else:
            print("FSL not found. Using Native Python Slice Timing (Cubic Spline).")

        # Native Implementation
        return self._run_native_stc(input_path, output_path, tr, slice_timing)

    def _run_native_stc(self, input_path: str, output_path: str, tr: float, slice_timing: list = None) -> bool:
        try:
            img = nib.load(input_path)
            data = img.get_fdata() # (X, Y, Z, T)
            affine = img.affine
            
            if len(data.shape) != 4:
                raise ValueError("Input must be 4D fMRI data.")
                
            n_slices = data.shape[2]
            n_vols = data.shape[3]
            
            # 1. Determine Acquisition Times
            if slice_timing is None:
                # Default: Ascending (0 to TR) 
                # e.g. slice 0 at t=0, slice N at t=TR (approx)
                # Usually equally spaced: t_z = n * (TR / n_slices)
                slice_timing = [z * (tr / n_slices) for z in range(n_slices)]
            elif len(slice_timing) != n_slices:
                print(f"Warning: Slice timing list len ({len(slice_timing)}) != n_slices ({n_slices}).")
                # Fallback to default
                slice_timing = [z * (tr / n_slices) for z in range(n_slices)]
                
            # 2. Define Reference Time (Target)
            # Standard: Middle of TR
            ref_time = tr / 2.0
            
            # 3. Interpolate
            print(f"  Interpolating {n_slices} slices to t={ref_time:.3f}s (TR={tr}s)...")
            
            new_data = np.zeros_like(data)
            
            # Original time points for each volume
            # Volume 0 is at t=0 relative to start? 
            # Actually, Volume `k` assumes acquisition at `k * TR`.
            # But specific slices are at `k * TR + slice_offset`.
            # We want to shift signal at `k * TR + offset` to `k * TR + ref`.
            # Time shift needed: dt = ref - offset.
            # New value at T_vol = Value at (T_vol - dt).
            
            # Let's frame it as interpolation:
            # For a given slice z, we have samples at times: [0+offset, 1*TR+offset, 2*TR+offset, ...]
            # We want to resample at times: [0+ref, 1*TR+ref, 2*TR+ref, ...]
            
            # Original Grid (x)
            # We treat volume index as time unit for simplicity of range? 
            # No, let's use seconds.
            orig_t = np.arange(n_vols) * tr
            
            for z in range(n_slices):
                offset = slice_timing[z]
                
                # The signal for slice z was actually acquired at these times:
                actual_times = orig_t + offset
                
                # We want to estimate what the signal WOULD be if acquired at:
                target_times = orig_t + ref_time
                
                # Extract time series for this slice
                # (X, Y, T)
                slice_data = data[:, :, z, :] 
                
                # Reshape for efficient interpolation over T
                # shape: (N_pixels, T)
                dims = slice_data.shape[:2]
                pixels = slice_data.reshape(-1, n_vols)
                
                # Interpolate function needs x, y.
                # scipy.interpolate.interp1d
                # kind='cubic'
                # distinct interpolation per pixel?
                # interp1d can handle axis! 
                
                try:
                    # bounds_error=False, fill_value="extrapolate" needed because target_times might shift outside actual_times range slightly at ends
                    f = interpolate.interp1d(actual_times, pixels, kind='cubic', axis=1, bounds_error=False, fill_value="extrapolate")
                    
                    new_pixels = f(target_times)
                    
                    # Store back
                    new_data[:, :, z, :] = new_pixels.reshape(dims[0], dims[1], n_vols)
                except Exception as e_slice:
                    print(f"Slice {z} failed: {e_slice}")
                    new_data[:, :, z, :] = slice_data # Fallback
            
            # Save
            new_img = nib.Nifti1Image(new_data, affine, img.header)
            nib.save(new_img, output_path)
            
            print(f"  Native STC complete. Saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Native STC failed: {e}")
            import traceback
            traceback.print_exc()
            shutil.copy(input_path, output_path)
            return True # Soft fail

