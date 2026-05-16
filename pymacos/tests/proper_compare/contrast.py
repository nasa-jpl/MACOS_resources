"""Coronagraph contrast metrics for macos<->PROPER tests.

The natural scoring for a coronagraph is the RADIALLY-AVERAGED
contrast vs angular separation in lambda/D units:

    contrast(r) = mean_intensity_in_radial_ring(r)  /  peak_unaberrated

normalised by the un-coronagraphed (no-mask) on-axis peak.  This is
the form coronagraph papers use; "dark-zone contrast" then refers to
the value of this curve over a working-angle range (e.g. 3-10 lambda/D).

This module provides:
  - radial_profile(image)        : 1D azimuthally-averaged profile
  - first_airy_null(psf)         : find first null of a centred PSF
  - lambda_over_D_pixels(psf)    : derive lambda/D in pixels from a PSF
  - radial_contrast(I, peak, ld) : full radial-contrast curve
  - plot_contrast_curves(...)    : multi-curve overlay PNG

The lambda/D conversion is derived EMPIRICALLY from an un-coronagraphed
PSF's first Airy null at 1.22 lambda/D.  That avoids having to know
the exact pupil diameter at the science focal plane (depends on the
full prescription's magnification chain).  Works as long as the
no-mask PSF is approximately Airy-like, which it is for any system
with a circularly-symmetric pupil + small aberrations.
"""
from __future__ import absolute_import

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Matplotlib in headless mode so the test runner doesn't open windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402


def radial_profile(image: np.ndarray,
                   center: Optional[Tuple[float, float]] = None,
                   max_radius: Optional[float] = None,
                   bin_size: float = 1.0,
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray]:
    """Azimuthally-averaged radial profile of a 2D image.

    Args:
        image: 2D array.
        center: (cy, cx) in pixel coordinates.  Defaults to the
            ``(N-1)/2`` array centre (FFT-shift convention for even N).
        max_radius: maximum radius to bin out to, in pixels.  Defaults
            to half the shorter image dimension.
        bin_size: bin width in pixels.  Default 1.

    Returns:
        (r_centers, mean, std, n) -- 1D arrays of the bin centres,
        mean value per bin, std-dev per bin, and pixel count per bin.
    """
    if center is None:
        cy = (image.shape[0] - 1) / 2.0
        cx = (image.shape[1] - 1) / 2.0
    else:
        cy, cx = center

    yy, xx = np.indices(image.shape)
    rr = np.hypot(yy - cy, xx - cx)

    if max_radius is None:
        max_radius = min(image.shape) / 2

    bins = np.arange(0.0, max_radius + bin_size, bin_size)
    n_bins = len(bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])

    means = np.full(n_bins, np.nan)
    stds  = np.full(n_bins, np.nan)
    ns    = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        msk = (rr >= bins[i]) & (rr < bins[i+1])
        if msk.any():
            vals = image[msk]
            means[i] = vals.mean()
            stds[i]  = vals.std()
            ns[i]    = vals.size
    # NaN-fill for empty bins (e.g., the r=[0, <1] bin under even-N
    # grids where no pixel sits at the (N-1)/2 array centre) keeps
    # log-y plotting clean (matplotlib skips NaN) and signals to
    # callers that the bin has no data rather than zero intensity.
    return centers, means, stds, ns


def first_airy_null(intensity: np.ndarray,
                    search_min_px: float = 3.0,
                    search_max_px: float = 60.0,
                    bin_size: float = 1.0,
                    null_max_fraction_of_peak: float = 0.05,
                    ) -> Optional[float]:
    """Estimate the first Airy null radius (pixels) from a centred PSF.

    Walks outward from the centre of the radial profile and returns
    the radius of the first interior local minimum whose value is
    below ``null_max_fraction_of_peak`` times the central peak.  The
    fractional-depth guard excludes spurious sub-pixel local minima
    that sit on the steep falling slope of the central Airy peak --
    those have values comparable to the peak and aren't real nulls.

    Returns ``None`` if no qualifying null is found in
    ``[search_min_px, search_max_px]``.
    """
    r, mean, _, _ = radial_profile(intensity, max_radius=search_max_px,
                                   bin_size=bin_size)
    peak = np.nanmax(mean)
    if peak <= 0 or not np.isfinite(peak):
        return None
    threshold = peak * null_max_fraction_of_peak

    for i in range(1, len(mean) - 1):
        if not (np.isfinite(mean[i - 1])
                and np.isfinite(mean[i])
                and np.isfinite(mean[i + 1])):
            continue
        if r[i] < search_min_px:
            continue
        if (mean[i] < mean[i - 1]
                and mean[i] < mean[i + 1]
                and mean[i] < threshold):
            return float(r[i])
    return None


def lambda_over_D_pixels(unaberrated_psf: np.ndarray,
                         search_min_px: float = 3.0,
                         search_max_px: float = 40.0,
                         ) -> float:
    """Derive lambda/D in pixel units from an un-coronagraphed PSF.

    First Airy null is at 1.22 lambda/D for a circular pupil.  Useful
    when the prescription's effective pupil diameter at the science
    focal plane isn't trivial to compute analytically.
    """
    r_null = first_airy_null(unaberrated_psf,
                             search_min_px=search_min_px,
                             search_max_px=search_max_px)
    if r_null is None:
        raise ValueError(
            "lambda_over_D_pixels: no Airy null found in "
            f"radius range [{search_min_px}, {search_max_px}] px -- "
            "check that the input is a centred un-coronagraphed PSF")
    return r_null / 1.22


def radial_contrast(intensity: np.ndarray,
                    peak_unaberrated: float,
                    lam_over_D_px: float,
                    max_lambda_over_D: float = 20.0,
                    bins_per_lambda_over_D: int = 4,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Radially-averaged contrast vs separation in lambda/D.

    Args:
        intensity: (N, N) focal-plane intensity to score.
        peak_unaberrated: peak of the un-coronagraphed reference PSF
            (Strehl normaliser).
        lam_over_D_px: lambda/D in pixels at this focal plane.
        max_lambda_over_D: how far out to score.
        bins_per_lambda_over_D: radial bin density.

    Returns:
        (r_lambda_over_D, contrast)
            r_lambda_over_D: separation in lambda/D
            contrast: mean(intensity in ring) / peak_unaberrated
    """
    bin_size_px   = lam_over_D_px / bins_per_lambda_over_D
    max_radius_px = max_lambda_over_D * lam_over_D_px
    r_px, mean, _, _ = radial_profile(intensity,
                                      max_radius=max_radius_px,
                                      bin_size=bin_size_px)
    return r_px / lam_over_D_px, mean / peak_unaberrated


def plot_contrast_curves(curves: dict,
                         out_path: Path,
                         title: str = "",
                         floor: float = 1e-18,
                         xlim: Optional[Tuple[float, float]] = None,
                         ylim: Optional[Tuple[float, float]] = None,
                         ) -> None:
    """Multi-curve contrast plot, log y, x in lambda/D.

    Args:
        curves: dict mapping legend label to (r_lambda_over_D, contrast).
        out_path: where to write the PNG.
        floor: contrast values < floor are clipped to floor for the
            log-y display (avoids log(0) / negative values from finite-
            precision arithmetic on suppressed-PSF arrays).
        xlim, ylim: axis limits; auto if None.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (r, c) in curves.items():
        ax.semilogy(r, np.maximum(c, floor), label=label, lw=1.5)
    ax.set_xlabel(r"separation ($\lambda/D$)")
    ax.set_ylabel("radial contrast (mean intensity / peak unaberrated)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
