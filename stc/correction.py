import numpy as np
from scipy.fft import fft, ifft, fftfreq


class SliceTimer:
    """
    SPM-style Slice Timing Correction

    Fixes applied vs original:
    - dt formula corrected: TA/Z  (not TA/(Z-1))
    - Hanning window removed from data (was distorting amplitude)
      SPM applies windowing to sinc kernel, NOT to the data signal
    - Phase shift sign convention made explicit and correct
    - ref_slice validated as 0-based index consistently
    - Added zero-shift early exit for ref_slice itself
    """

    def __init__(
        self,
        data,
        tr,
        slice_order=None,
        ref_slice=0,
        ta=None,
        slice_times=None
    ):
        self.data   = data.astype(np.float32)
        self.tr     = float(tr)
        self.slice_order = slice_order
        self.ref_slice   = ref_slice
        self.ta          = ta
        self.external_slice_times = slice_times

        self.X, self.Y, self.Z, self.T = self.data.shape

        if not (0 <= ref_slice < self.Z):
            raise ValueError(
                f"ref_slice={ref_slice} out of range [0, {self.Z - 1}]"
            )

        # ----------------------------------------------------------
        # TA — SPM default: TR - (TR / nSlices)
        # ----------------------------------------------------------
        if self.ta is None:
            self.ta = self.tr - (self.tr / self.Z)

        # ----------------------------------------------------------
        # FIX 1: dt = TA / Z   (SPM uses Z, not Z-1)
        #
        # Why Z and not Z-1?
        # SPM treats each inter-slice gap as TA/Z.
        # Using Z-1 would only cover gaps between slices (fencepost error),
        # but SPM counts the gap after the last slice too (volume spacing).
        # ----------------------------------------------------------
        self.dt = self.ta / self.Z if self.Z > 1 else 0.0

        # ----------------------------------------------------------
        # Validate fallback slice_order
        # ----------------------------------------------------------
        if self.external_slice_times is None:
            if self.slice_order is None:
                raise ValueError(
                    "Either slice_times (from JSON) or slice_order must be provided"
                )
            if len(self.slice_order) != self.Z:
                raise ValueError(
                    f"slice_order length {len(self.slice_order)} != nSlices {self.Z}"
                )

    # ----------------------------------------------------------
    # Compute slice acquisition times (seconds)
    # ----------------------------------------------------------

    def _compute_slice_times(self):
        """
        Returns a 1D array of length Z where:
            slice_times[i] = time (in seconds) at which slice i was acquired
                             relative to the start of the volume
        """

        # JSON SliceTiming → use directly (most accurate)
        if self.external_slice_times is not None:
            if len(self.external_slice_times) != self.Z:
                raise ValueError(
                    f"SliceTiming length {len(self.external_slice_times)} != nSlices {self.Z}"
                )
            return np.array(self.external_slice_times, dtype=np.float32)

        # Compute from slice_order using corrected dt = TA/Z
        slice_times = np.zeros(self.Z, dtype=np.float32)
        for acquisition_index, slice_index in enumerate(self.slice_order):
            slice_times[slice_index] = acquisition_index * self.dt

        return slice_times

    # ----------------------------------------------------------
    # Fourier phase shift — SPM-style sinc interpolation
    # ----------------------------------------------------------

    def _fourier_shift_slice(self, slice_data, shift_seconds):
        """
        Shifts the BOLD time series of one slice by shift_seconds
        using Fourier phase shift theorem.

        shift_seconds > 0 → shift signal forward in time
        shift_seconds < 0 → shift signal backward in time

        FIX 2: Hanning window REMOVED from data.
        Original code applied np.hanning(T) directly to the BOLD signal
        before FFT. This distorts signal amplitude — the Hanning taper
        is meant to smooth the sinc interpolation kernel, not the data.
        SPM does NOT window the data signal itself.
        """

        # Remove voxel-wise mean (SPM behavior — centers signal at 0)
        mean_vals  = np.mean(slice_data, axis=2, keepdims=True)
        slice_data = slice_data - mean_vals

        # FFT along time axis (axis=2 → T dimension)
        F = fft(slice_data, axis=2)

        # Frequency bins in Hz
        freqs = fftfreq(self.T, d=self.tr)   # shape: (T,)
        freqs = freqs.reshape(1, 1, self.T)  # broadcast over X, Y

        # ----------------------------------------------------------
        # FIX 3: Phase shift sign — made explicit
        #
        # To shift a signal FORWARD by shift_seconds:
        #   phase = exp(-2j * pi * f * shift_seconds)
        #
        # Our shift_seconds = ref_time - acq_time
        # So phase = exp(-2j*pi*f*(ref-acq))
        #          = exp(+2j*pi*f*(acq-ref))
        # This correctly advances slices acquired BEFORE ref_time
        # and delays slices acquired AFTER ref_time.
        # ----------------------------------------------------------
        phase   = np.exp(-2j * np.pi * freqs * shift_seconds)
        shifted = ifft(F * phase, axis=2).real

        # Restore mean
        shifted = shifted + mean_vals

        return shifted.astype(np.float32)

    # ----------------------------------------------------------
    # Main correction loop
    # ----------------------------------------------------------

    def correct(self):
        """
        Run STC on all slices.
        Returns corrected 4D array, same shape as input.
        """

        slice_times = self._compute_slice_times()
        ref_time    = slice_times[self.ref_slice]

        corrected = np.zeros_like(self.data, dtype=np.float32)

        for z in range(self.Z):

            shift = ref_time - slice_times[z]

            # Early exit: no correction needed for reference slice
            if shift == 0.0:
                corrected[:, :, z, :] = self.data[:, :, z, :]
                continue

            corrected[:, :, z, :] = self._fourier_shift_slice(
                self.data[:, :, z, :],
                shift
            )

        return corrected