# -*- coding: utf-8 -*-
"""
segment_mrf.py
══════════════════════════════════════════════════════════════════
Markov Random Field (MRF) cleanup

Matches SPM batch:
  MRF Parameter : 1     (strength of spatial regularisation)
  Clean Up      : Light clean  (removes isolated voxels)

SPM's MRF is a mean-field approximation that encourages spatial
smoothness in the tissue label field by incorporating neighbourhood
information into the posterior update.
══════════════════════════════════════════════════════════════════
"""

import sys, io
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label as ndlabel

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# 6-connected neighbourhood (faces only — matches SPM MRF connectivity)
FACE_CONNECT = np.array([
    [[0,0,0],[0,1,0],[0,0,0]],
    [[0,1,0],[1,0,1],[0,1,0]],
    [[0,0,0],[0,1,0],[0,0,0]],
], dtype=np.float64)
FACE_CONNECT[1,1,1] = 0   # exclude centre


class MRFCleanup:
    """
    Mean-field MRF update + light morphological cleanup.

    Steps (matching SPM spm_mrf.m):
    1. MRF smoothing: update each voxel's posterior using its
       6-connected neighbours' class probabilities
    2. Light morphological cleanup: erode then dilate GM/WM/CSF
       to remove isolated floating voxels
    """

    def __init__(self,
                 mrf_strength: float = 1.0,
                 n_mrf_iter:   int   = 3):
        """
        Parameters
        ----------
        mrf_strength : SPM batch 'MRF Parameter' = 1
        n_mrf_iter   : number of MRF mean-field iterations
        """
        self.mrf_strength = mrf_strength
        self.n_mrf_iter   = n_mrf_iter

    # ── MRF mean-field update ─────────────────────────────────────────

    def apply_mrf(self,
                  posteriors: np.ndarray,
                  verbose:    bool = True) -> np.ndarray:
        """
        Apply MRF spatial smoothing to tissue posteriors.

        posteriors : (X,Y,Z,K) float32
        Returns    : (X,Y,Z,K) float32  updated posteriors
        """
        from scipy.ndimage import uniform_filter

        if verbose:
            print("  Applying MRF spatial regularisation...")

        n_class  = posteriors.shape[3]
        updated  = posteriors.copy().astype(np.float64)

        # MRF potential: beta * sum of neighbour posteriors for same class
        beta = self.mrf_strength

        for it in range(self.n_mrf_iter):
            neighbour_sum = np.zeros_like(updated)

            for k in range(n_class):
                # Convolve class-k map with face-connectivity kernel
                # → each voxel gets the sum of its 6 neighbours' posteriors
                from scipy.ndimage import convolve
                neighbour_sum[..., k] = convolve(
                    updated[..., k],
                    FACE_CONNECT,
                    mode="constant",
                    cval=0.0,
                )

            # MRF-modulated posteriors:
            # new_post[k] ∝ old_post[k] * exp(beta * neighbour_sum[k])
            log_mrf = beta * neighbour_sum
            # Subtract max per voxel for numerical stability
            log_mrf -= log_mrf.max(axis=3, keepdims=True)
            mrf_weight = np.exp(log_mrf)

            updated = updated * mrf_weight
            # Renormalise
            total   = updated.sum(axis=3, keepdims=True)
            total   = np.where(total < 1e-10, 1.0, total)
            updated = updated / total

        if verbose:
            print(f"    MRF done ({self.n_mrf_iter} iterations, beta={beta})")

        return updated.astype(np.float32)

    # ── Light morphological cleanup ───────────────────────────────────

    def light_cleanup(self,
                      posteriors: np.ndarray,
                      verbose:    bool = True) -> np.ndarray:
        """
        SPM 'Light clean': erode then re-dilate GM/WM/CSF maps to
        remove small isolated clusters.

        Operates on the hard label map (argmax) then redistributes
        probabilities accordingly.
        """
        if verbose:
            print("  Applying light morphological cleanup...")

        # Hard labels for GM (0), WM (1), CSF (2)
        gm_soft  = posteriors[..., 0]
        wm_soft  = posteriors[..., 1]
        csf_soft = posteriors[..., 2]

        struct = np.ones((3, 3, 3), dtype=bool)   # 26-connected

        def _clean_map(soft_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
            """Remove voxels not connected to the main component."""
            binary = soft_map > threshold
            if not binary.any():
                return soft_map

            # Label connected components
            labelled, n_comp = ndlabel(binary, structure=struct)
            if n_comp <= 1:
                return soft_map

            # Keep only the largest component
            sizes    = np.bincount(labelled.ravel())
            sizes[0] = 0   # ignore background label
            largest  = sizes.argmax()
            mask     = labelled == largest

            # Zero out non-connected voxels in the soft map
            cleaned  = soft_map.copy()
            cleaned[~mask] = 0.0
            return cleaned

        clean = posteriors.copy()
        clean[..., 0] = _clean_map(gm_soft)
        clean[..., 1] = _clean_map(wm_soft)
        clean[..., 2] = _clean_map(csf_soft)

        # Renormalise after cleanup
        total = clean.sum(axis=3, keepdims=True)
        total = np.where(total < 1e-10, 1.0, total)
        clean = clean / total

        if verbose:
            n_removed = int((posteriors[..., :3].argmax(3) != clean[..., :3].argmax(3)).sum())
            print(f"    Cleanup relabelled ~{n_removed:,} voxels")

        return clean.astype(np.float32)

    # ── Combined pipeline ─────────────────────────────────────────────

    def run(self,
            posteriors: np.ndarray,
            verbose:    bool = True) -> np.ndarray:
        """Run MRF smoothing then light cleanup (SPM order)."""
        p = self.apply_mrf(posteriors, verbose=verbose)
        p = self.light_cleanup(p, verbose=verbose)
        return p