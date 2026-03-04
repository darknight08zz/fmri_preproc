import numpy as np
from scipy.fft import fft, ifft, fftfreq


class SliceTimer:
    """
    SPM-style Slice Timing Correction

    Features:
    - Uses external SliceTiming vector if provided (from JSON)
    - Otherwise computes slice timing from slice_order + TA
    - Proper TA handling (SPM default if not provided)
    - Fourier phase shift (true sinc interpolation)
    - Mean removal and restoration
    - Hanning taper to reduce edge ringing
    - Vectorized across X,Y for performance
    """

    # ----------------------------------------------------------
    # Constructor
    # ----------------------------------------------------------

    def __init__(
        self,
        data,
        tr,
        slice_order=None,
        ref_slice=0,
        ta=None,
        slice_times=None
    ):

        self.data = data.astype(np.float32)
        self.tr = float(tr)
        self.slice_order = slice_order
        self.ref_slice = ref_slice
        self.ta = ta
        self.external_slice_times = slice_times

        self.X, self.Y, self.Z, self.T = self.data.shape

        if not (0 <= ref_slice < self.Z):
            raise ValueError("Reference slice out of range")

        # ------------------------------------------------------
        # TA handling (SPM default)
        # ------------------------------------------------------

        if self.ta is None:
            # SPM default:
            # TA = TR - (TR / Nslices)
            self.ta = self.tr - (self.tr / self.Z)

        if self.Z > 1:
            self.dt = self.ta / (self.Z - 1)
        else:
            self.dt = 0.0

        # ------------------------------------------------------
        # Validate fallback slice_order
        # ------------------------------------------------------

        if self.external_slice_times is None:
            if self.slice_order is None:
                raise ValueError(
                    "Either slice_times (JSON) or slice_order must be provided"
                )

            if len(self.slice_order) != self.Z:
                raise ValueError(
                    "slice_order length must equal number of slices"
                )

    # ----------------------------------------------------------
    # Compute slice acquisition times
    # ----------------------------------------------------------

    def _compute_slice_times(self):

        # If JSON SliceTiming provided → use directly
        if self.external_slice_times is not None:

            if len(self.external_slice_times) != self.Z:
                raise ValueError(
                    "SliceTiming vector length does not match number of slices"
                )

            return np.array(self.external_slice_times, dtype=np.float32)

        # Otherwise compute from slice_order
        slice_times = np.zeros(self.Z, dtype=np.float32)

        for acquisition_index, slice_index in enumerate(self.slice_order):
            slice_times[slice_index] = acquisition_index * self.dt

        return slice_times

    # ----------------------------------------------------------
    # Fourier-based temporal shift (SPM-style)
    # ----------------------------------------------------------

    def _fourier_shift_slice(self, slice_data, shift_seconds):

        # Remove voxel-wise mean (SPM behavior)
        mean_vals = np.mean(slice_data, axis=2, keepdims=True)
        slice_data = slice_data - mean_vals

        # Apply Hanning taper to reduce edge artifacts
        window = np.hanning(self.T).reshape(1, 1, self.T)
        slice_data = slice_data * window

        # FFT along time axis
        f = fft(slice_data, axis=2)

        # Frequencies in Hz
        freqs = fftfreq(self.T, d=self.tr)

        # Compute phase shift
        phase = np.exp(-2j * np.pi * freqs * shift_seconds)
        phase = phase.reshape(1, 1, self.T)

        # Apply phase shift
        shifted = ifft(f * phase, axis=2).real

        # Restore mean
        shifted += mean_vals

        return shifted.astype(np.float32)

    # ----------------------------------------------------------
    # Main Correction Routine
    # ----------------------------------------------------------

    def correct(self):

        slice_times = self._compute_slice_times()
        ref_time = slice_times[self.ref_slice]

        corrected = np.zeros_like(self.data, dtype=np.float32)

        for z in range(self.Z):

            shift = ref_time - slice_times[z]

            slice_data = self.data[:, :, z, :]
            corrected[:, :, z, :] = self._fourier_shift_slice(
                slice_data,
                shift
            )

        return corrected
