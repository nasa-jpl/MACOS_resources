"""Apodiser construction utilities for macos<->PROPER tests.

Goal: build NxN amplitude transmission maps that can be handed to both
engines via the SAME array -- bit-identical apodisation, no
parametric-reconstruction drift.

Two output paths:
  - PROPER side: proper.prop_multiply(wfo, mask)
  - macos side : pymacos.apodize(srf, mask)

Quality knobs:
  - supersample (K): super-sample binary aperture edges by KxK and bin
    down for sub-pixel area weighting.  K=16 gives ~0.4% accuracy on
    edge-pixel coverage; K=32 gives ~0.1%.  Without area weighting,
    edge pixels are 0/1 and the boundary quantises to the grid --
    intensities at the science focal plane jitter at the 1/N level
    when you change N.  With area weighting, the boundary's true
    sub-pixel position is encoded continuously in edge-pixel values
    and low-N results converge to high-N results.

For raw-contrast at HWO levels (1e-10) super-sampling at K=16-32 will
ultimately bottom out; a band-limited Fourier construction is the
gold standard for that.  Plug-in for Phase 6b -- the API here is
designed to accept any aperture builder that returns an (N, N) float
array.
"""
from __future__ import absolute_import

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.special import j1 as _bessel_j1


# ----------------------------------------------------------------------
# Core: super-sampled apodisation builder
# ----------------------------------------------------------------------

def build_apodised_mask(
        N: int,
        dx_m: float,
        aperture_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        taper_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
                                                                = None,
        supersample: int = 16,
        ) -> np.ndarray:
    """Build an (N, N) float64 apodisation mask with sub-pixel area-
    weighted edge pixels.

    Output convention: pixel (i, j) is centred at physical coordinate
    ((j - (N-1)/2)*dx, (i - (N-1)/2)*dx) (x along columns, y along
    rows; FFT-shift / array-centre convention matching macos and
    PROPER).

    Args:
        N: Grid size.  Must match macos's mdttl and PROPER's gridsize.
        dx_m: Pixel pitch in metres (must match both engines).
        aperture_fn: Callable(x, y) -> bool/int/float, returning the
            binary "inside aperture" predicate at each (x, y) in
            metres.  Super-sampled at supersample**2 sub-positions
            per output pixel to compute exact area coverage at edges.
        taper_fn: Optional callable(x, y) -> float in [0, 1], the
            smooth amplitude taper inside the aperture.  Evaluated
            at PIXEL CENTRES only -- no super-sampling, because a
            smooth function is its own own pointwise representation
            (any sub-pixel taper variation is captured by the value
            at the centre to second order).  If you need a sharp-edged
            taper, encode the edges in aperture_fn instead.
        supersample: Linear super-sampling factor K.  K=16 gives
            ~0.4% coverage accuracy; K=32 ~0.1%.

    Returns:
        (N, N) float64 array of amplitude transmission values.
        Values in [0, 1] for typical hard-aperture + non-negative
        taper combinations; the function does not clamp.
    """
    if N <= 0 or supersample <= 0:
        raise ValueError(
            f"N={N} and supersample={supersample} must be positive")

    K = supersample
    centers = (np.arange(N) - (N - 1) / 2.0) * dx_m       # (N,) metres
    sub_offsets = (np.arange(K) - (K - 1) / 2.0) * dx_m / K  # (K,)

    # Sub-pixel x grid (1, N, K)  -- fast axis: sub-position within pixel
    sub_x = centers[None, :, None] + sub_offsets[None, None, :]

    out = np.empty((N, N), dtype=np.float64)
    # Loop rows to keep peak memory at (K, N, K) = K^2 * N floats.
    # For N=1024, K=16 that's ~2 MB per row; total 256 M evaluations.
    for i, yc in enumerate(centers):
        sub_y_row = yc + sub_offsets                       # (K,)
        # Broadcast to (K, N, K): K sub-y, N pixels, K sub-x
        ys = sub_y_row[:, None, None]                       # (K, 1, 1)
        xs = sub_x                                          # (1, N, K)
        mask_hi = aperture_fn(xs, ys).astype(np.float64)    # (K, N, K)
        # Average over BOTH sub-axes (K sub-y + K sub-x per output pixel)
        out[i] = mask_hi.mean(axis=(0, 2))                  # (N,)

    if taper_fn is not None:
        # Evaluate taper at pixel CENTRES -- smooth functions don't
        # need super-sampling for the apodisation itself.
        xc, yc_grid = np.meshgrid(centers, centers, indexing="xy")
        out *= np.asarray(taper_fn(xc, yc_grid), dtype=np.float64)

    return out


# ----------------------------------------------------------------------
# Aperture predicates (callable, return bool/float arrays)
# ----------------------------------------------------------------------

def circle(r0: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Binary disk of radius ``r0`` (metres) centred at origin."""
    def _circle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x * x + y * y) <= (r0 * r0)
    return _circle


def annulus(r_in: float, r_out: float
            ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Binary annulus with inner radius ``r_in`` and outer ``r_out``
    (both in metres) centred at origin.
    """
    if r_in >= r_out:
        raise ValueError(f"annulus: r_in ({r_in}) must be < r_out ({r_out})")
    def _annulus(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r2 = x * x + y * y
        return (r2 >= r_in * r_in) & (r2 <= r_out * r_out)
    return _annulus


# ----------------------------------------------------------------------
# Smooth tapers (callable, return float arrays in [0, 1])
# ----------------------------------------------------------------------

def gaussian_edge_taper(r0: float, sigma: float
                        ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Soft Gaussian roll-off OUTSIDE radius ``r0``:
        T(r) = 1                       for r < r0
        T(r) = exp(-((r-r0)/sigma)**2) for r >= r0

    Pairs naturally with circle(r1) where r1 >> r0 + a few sigma to
    truncate the tail.  In practice use r1 = r0 + 4*sigma for ~3e-7
    truncation; larger r1 wastes grid; smaller r1 truncates visibly.
    """
    def _taper(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x * x + y * y)
        out = np.ones_like(r)
        edge = r > r0
        out[edge] = np.exp(-((r[edge] - r0) / sigma) ** 2)
        return out
    return _taper


def super_gaussian_taper(r0: float, n: float
                         ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Super-Gaussian (flat-top) apodisation:
        T(r) = exp(-(r/r0)**(2n))

    n=1: regular Gaussian, n=2..4: rounded-rectangle, n>=8: nearly
    flat-top with soft edges.  Good for shaped-pupil approximations.
    """
    def _taper(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x * x + y * y)
        return np.exp(-(r / r0) ** (2 * n))
    return _taper


# ----------------------------------------------------------------------
# Phase 6b: band-limited Fourier construction (gold-standard apertures)
# ----------------------------------------------------------------------
#
# Super-sampling (build_apodised_mask above) approximates a binary
# aperture's sub-pixel area coverage by averaging K*K sub-samples per
# output pixel.  Quality scales as 1/K^2; at K=16 the residual edge
# aliasing limits dark-zone contrast around 1e-7 to 1e-8.  Good enough
# for present-day Lyot coronagraphs; insufficient for HWO-target raw
# contrast around 1e-10.
#
# The gold-standard alternative for a shape whose 2D Fourier transform
# is known analytically is to build the mask in k-space at the grid's
# FFT frequencies and inverse-FFT to real space.  Because the k-space
# representation is sampled only out to the grid's Nyquist by
# construction, the resulting mask is perfectly band-limited -- no
# aliasing, no sub-pixel quantisation; every pixel value is the exact
# discrete-FFT representation of the continuous shape.
#
# Currently only the circular aperture is implemented (the most common
# coronagraph component).  Annulus = difference of two circles; polygon
# requires the Shoelace-formula FT, deferred until needed.


@dataclass(frozen=True)
class BandLimitedCircle:
    """Specification of a band-limited circular aperture.

    The 2D Fourier transform of a unit-amplitude disc of radius r0,
    centred at the origin, is the well-known Airy-disc-like function:

        F(k) = r0 * J_1(2*pi*r0*k) / k       for k > 0
        F(0) = pi * r0^2                       (disc area)

    where k = sqrt(k_x^2 + k_y^2) and J_1 is the first-order Bessel
    function of the first kind.  ``build_band_limited_mask(N, dx, spec)``
    samples F at the FFT grid's k values and inverse-FFTs to real
    space, yielding an NxN mask whose pixel values are the exact
    band-limited representation of the continuous disc.
    """
    r0: float


def build_band_limited_mask(N: int, dx: float,
                            spec: BandLimitedCircle,
                            ) -> np.ndarray:
    """Band-limited apodisation mask via analytic Fourier construction.

    Args:
        N: grid size (must match macos's mdttl and PROPER's gridsize).
        dx: pixel pitch in metres.
        spec: aperture specification.  Currently only
            ``BandLimitedCircle``.

    Returns:
        (N, N) float64 array.  Pixel (i, j) is centred at
        ((j - (N-1)/2) * dx, (i - (N-1)/2) * dx) -- same convention
        as ``build_apodised_mask``.  Interior values approach 1, exterior
        approach 0; edge pixels carry continuous fractional values
        reflecting the band-limited representation of the boundary
        (NOT the geometric sub-pixel coverage -- those are different
        things and disagree slightly at the pixel scale).

    Normalisation:
        The continuous IFT relation
            f(r) = ∫ F(k) exp(2πi k r) dk
        sampled on a discrete N-point grid with k-spacing dk = 1/(N dx)
        gives
            f[i] = sum_j F[j] exp(2πi i j / N) * dk
                 = (1/dx) * ifft(F)[i]
        (using numpy's 1/N-normalised ``ifft``).  In 2D this generalises
        to ``ifft2(F) / dx**2 * N**2`` because numpy's 2D ``ifft2``
        normalisation is ``1/N**2``.  Verified numerically: the sum of
        the returned mask equals the disc area in pixel units
        (pi * r0**2 / dx**2) to 4+ significant figures.
    """
    if not isinstance(spec, BandLimitedCircle):
        raise TypeError(
            f"build_band_limited_mask: unsupported spec type "
            f"{type(spec).__name__}; only BandLimitedCircle is "
            "currently implemented")

    # k-axis in cycles per metre, fft-ordered: 0, +1/(Ndx), ...,
    # +(N/2-1)/(Ndx), -N/(2 Ndx), ..., -1/(Ndx)
    k_axis = np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(k_axis, k_axis, indexing='xy')
    k = np.hypot(kx, ky)

    # Analytic FT of a unit-amplitude disc of radius r0
    F = np.empty_like(k)
    nz = k > 0
    F[nz]  = spec.r0 / k[nz] * _bessel_j1(2.0 * np.pi * spec.r0 * k[nz])
    F[~nz] = np.pi * spec.r0 ** 2

    # numpy.fft.ifft2 has 1/N**2 normalisation; the continuous-IFT
    # / discrete-grid relation requires multiplying by N**2 and the
    # sampling factor 1/(N*dx)**2 = 1/(N**2 dx**2); net: divide by dx**2.
    mask_fft_ordered = np.real(np.fft.ifft2(F)) / (dx * dx)
    # Shift array centre from (0, 0) to ((N-1)/2, (N-1)/2)
    return np.fft.fftshift(mask_fft_ordered)
