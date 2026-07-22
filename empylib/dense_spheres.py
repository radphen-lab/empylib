# -*- coding: utf-8 -*-
"""
Dense-sphere Percus-Yevick structure factors for hard-sphere ensembles.

Implements Botet, Kwok & Cabane (2020), "Percus-Yevick structure factors
made simple" (J. Appl. Cryst. 53), for monodisperse and polydisperse
hard-sphere systems, plus closed-form radius-distribution kernels (Schulz,
truncated normal, inverse Gaussian, exponential) that bypass numerical
integration entirely -- useful for repeated evaluation inside
optimization / inverse-design loops.
"""
import warnings as _warnings
from dataclasses import dataclass as _dataclass, field as _field
from typing import List as _List, Union as _Union

import numpy as _np

from .utils import _check_mie_inputs, _hide_signature

__all__ = (
    'structure_factor_PY',
    'SchulzDistribution',
    'TruncatedNormalDistribution',
    'InverseGaussianDistribution',
    'ExponentialDistribution',
    'schulz',
    'truncated_normal',
    'inverse_gaussian',
    'exponential',
)


def _trapz(y, x, axis=-1):
    """Compat wrapper for NumPy trapezoidal integration across versions."""
    if hasattr(_np, "trapezoid"):
        return _np.trapezoid(y, x, axis=axis)
    return _np.trapz(y, x, axis=axis)


def _py_assemble_Sq(b, c, d, e, f, g):
    """
    Assemble the Percus-Yevick structure factor S(q) from Botet et al.'s six
    real auxiliary functions (Eqs. 4-6). Shared by the trapz-integrated
    (tabulated) kernel and every closed-form distribution kernel below.
    """
    denom = d**2 + e**2
    X = 1 + b + (2*e*f*g + d*(f**2 - g**2)) / denom
    Y = c + (2*d*f*g - e*(f**2 - g**2)) / denom
    return (Y / c) / (X**2 + Y**2)


def _mono_percus_yevick(fv, q, D):
    """
    Compute the Percus-Yevick structure factor S(q) for monodispersed
    hard-sphere systems.

    References: Kinning, D. J., & Thomas, E. L. (1984).
                Hard-Sphere Interactions between Spherical Domains in Diblock Copolymers.
                Macromolecules, 17(9), 1712-1718.

    Radius convention: a = D/2 (Botet's radius), consistent with the
    polydisperse kernel below.

    Parameters:
    -----------
    fv : float
        Volume fraction (phi) of the spheres.
    q : float
        Magnitude of the scattering vector.
    D : float
        Diameter of the sphere.

    Returns:
    --------
    S_q : float
        Structure factor evaluated at q.
    """
    if isinstance(D, _np.ndarray):
        assert D.size == 1, "For monodisperse case, D must be a single value."
        D = D.item()

    if not isinstance(D, float) and not isinstance(D, int):
        raise ValueError("For monodisperse case, D must be a float or int.")

    R = D / 2  # Botet's radius convention: a = D/2
    x = 2 * q * R  # Scattering variable as defined by Kinning & Thomas
    # Guard against q == 0 exactly (e.g. theta=0, forward scattering): G_kt/x
    # below has a well-defined finite limit as x->0, but each term of G_kt
    # divides by x**2/x**3/x**5, so x==0 exactly gives 0/0 -> nan rather than
    # the limit. A tiny floor keeps every term a division of tiny-but-nonzero
    # numbers, which converges numerically to the correct limit.
    x = _np.where(_np.abs(x) < 1e-8, 1e-8, x)

    # Coefficients from Eq. (17)
    alpha = (1 + 2 * fv)**2 / (1 - fv)**4
    beta = -6 * fv * (1 + fv / 2)**2 / (1 - fv)**4
    gamma = 0.5 * fv * (1 + 2 * fv)**2 / (1 - fv)**4

    # G(A) from Eq. (21)
    term1 = alpha / x**2 * (_np.sin(x) - x * _np.cos(x))
    term2 = beta / x**3 * (2 * x * _np.sin(x) + (2 - x**2) * _np.cos(x) - 2)
    term3 = gamma / x**5 * (-x**4 * _np.cos(x) +
                        4 * ((3 * x**2 - 6) * _np.cos(x) +
                                (x**3 - 6 * x) * _np.sin(x) + 6))
    G_kt = term1 + term2 + term3

    # Structure factor from Eq. (20)
    S_q = 1 / (1 + 24 * fv * G_kt / x)
    return S_q


def _poly_percus_yevick(fv, qq, D, nD):
    """
    Compute the Percus-Yevick structure factor S(q) for polydisperse
    hard-sphere systems via numerical (trapezoidal) integration over a
    tabulated radius distribution.

    `nD` must be a **number** distribution n(D) (Botet's n(a), a = D/2),
    not volume- or intensity-weighted. It need not be normalized -- ratios
    of averaged quantities cancel any overall scale.

    References: Botet, R., Kwok, R., & Cabane, B. (2020).
                Percus-Yevick structure factors made simple.
                Journal of Applied Crystallography, 53(6), 1526-1534.

    For `fv > 0.5`, an approximate "complementary Percus-Yevick" matter/void
    (Babinet) swap is applied. This is NOT an exact PY solution for
    `fv > 0.5` -- Botet's own Monte Carlo comparison shows agreement
    degrading beyond fv~0.5, and hard-sphere packing algorithms cap out
    near fv~0.69. Treat results for fv approaching or exceeding ~0.6 with
    caution.

    Parameters:
    -----------
    fv : float
        Volume fraction (phi) of the spheres.
    qq : ndarray
        Magnitude of the scattering vector.
    D : ndarray
        Diameter of the spheres
    nD : _np.ndarray or None
        Number distribution over D (same length as D). If None, assumes monodisperse.

    Returns:
    --------
    S_q : float
        Structure factor evaluated at q.
    """
    if not isinstance(D, _np.ndarray) or not isinstance(nD, _np.ndarray):
        raise ValueError("D and nD must be numpy arrays in the polydisperse case.")

    if D.shape != nD.shape:
        raise ValueError("D and nD must have the same shape.")

    qq = _np.asarray(qq, dtype=float)
    squeeze_out = False
    if qq.ndim == 1:
        qq = qq[None, :]
        squeeze_out = True
    elif qq.ndim != 2:
        raise ValueError("qq must be a 1D or 2D array.")

    R = D / 2  # Botet's radius convention: a = D/2

    # if fv > 0.5, compute structure factor for voids
    # "complementary PY hard-sphere approach" -- approximate, see docstring.
    if fv > 0.5:
        _warnings.warn(
            f"fv={fv:.3f} > 0.5 uses the approximate complementary-PY "
            "(matter/void Babinet) treatment; accuracy degrades beyond "
            "fv~0.5 (Botet et al. 2020).",
            UserWarning,
        )
        R = (1 - fv)/fv*R
        fv = 1 - fv

    # Guard against q == 0 exactly (e.g. theta=0, forward scattering): every
    # b,c,d,e,f,g below is a ratio of distribution-averaged quantities that
    # both vanish at matching order as q->0 (a removable singularity with a
    # well-defined finite limit), but at q==0 exactly both sides of every
    # ratio are the literal float 0, giving 0/0 = nan rather than the limit.
    # A tiny floor -- far smaller than the Fsc stabilization threshold below,
    # so it does not affect accuracy anywhere the ratio is already
    # well-conditioned -- keeps every ratio a division of tiny-but-nonzero
    # numbers, which converges numerically to the correct limit.
    qq = _np.where(_np.abs(qq) < 1e-8, 1e-8, qq)

    # Vectorized over wavelength and theta dimensions to avoid Python loops.
    # qq shape: (n_wavelengths, n_theta)
    x = qq[:, :, None] * R[None, None, :]  # (n_wavelengths, n_theta, n_bins)

    # Psi is an auxiliary prefactor: psi = 3*phi / (1 - phi)
    psi = 3 * fv / (1 - fv)

    nD_w = nD[None, None, :]
    den_x3 = _trapz((x**3) * nD_w, R, axis=2)

    # Trigonometric building blocks for structure factor (Botet et al., Eqs. 8-13).
    Fcs = _np.cos(x) + x * _np.sin(x)  # cos(x) + x*sin(x); no cancellation, no guard needed

    # Fsc = sin(x) - x*cos(x) is a difference of two O(1) terms whose result
    # is O(x**3): direct evaluation suffers catastrophic cancellation for
    # small x. Every b,c,d,e,f,g below is a ratio of distribution-averaged
    # quantities that vanish at matching order as x->0 (a removable
    # singularity, not a true divergence), so we stabilize Fsc with its
    # Taylor series near x=0 rather than clamping q away from 0.
    x_thresh = 1e-2
    Fsc_series = x**3 / 3 * (1 - x**2/10 + x**4/280)  # sin(x) - x*cos(x), stable near 0
    Fsc_direct = _np.sin(x) - x * _np.cos(x)
    Fsc = _np.where(_np.abs(x) < x_thresh, Fsc_series, Fsc_direct)

    # Weighted averages over radius axis (distribution integral for each
    # wavelength/theta pair). This is the polydisperse equivalent of scalar
    # moments in the monodisperse PY formulas.
    avg = lambda f: _trapz(f * nD_w, R, axis=2)

    # Botet et al. expressions for b, c, d, e, f, g
    b = psi * avg(Fcs * Fsc) / den_x3
    c = psi * avg(Fsc**2) / den_x3
    d = 1 + psi * avg(x**2 * _np.sin(x) * _np.cos(x)) / den_x3
    e = psi * avg(x**2 * _np.sin(x)**2) / den_x3
    f = psi * avg(x * _np.sin(x) * Fsc) / den_x3
    g = - psi * avg(x * _np.cos(x) * Fsc) / den_x3

    S_q = _py_assemble_Sq(b, c, d, e, f, g)
    if squeeze_out:
        return S_q[0]
    return S_q


@_hide_signature
def structure_factor_PY(wavelength: _Union[float, _np.ndarray],
                        Nh: _Union[float, _np.ndarray],
                        D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]], None],
                        fv: float,
                        *,
                        theta: _Union[float, _np.ndarray] = None,
                        size_dist=None,
                        check_inputs: bool = True):
    """
    Compute the Percus-Yevick structure factor S(q) for hard-sphere systems,
    for monodisperse, tabulated-polydisperse, and closed-form-polydisperse
    cases.

    Parameters:
    -----------
    wavelength : ndarray or float
        Wavelengtgh (microns)

    Nh : ndarray or float
        Complex refractive index of host. If ndarray, len(Nh) == len(wavelength)

    D : float, ndarray, or None
        Diameter of the spheres. Use float for monodisperse, or array for
        tabulated polydisperse. May be None when `size_dist` is a
        closed-form distribution object (schulz(), truncated_normal(),
        inverse_gaussian(), exponential()) -- the analytic kernel does not
        need a diameter grid.

    fv : float
        Volume fraction (phi) of the spheres.

    theta : float or ndarray (optional)
        Scattering angle (radians). Default None

    size_dist : ndarray or closed-form distribution object (optional)
        Diameter **number**-density distribution. Either:
          - a tabulated ndarray with len(size_dist) == len(D), integrated
            numerically (as before), or
          - a closed-form distribution object built by schulz(),
            truncated_normal(), inverse_gaussian(), or exponential(),
            evaluated analytically (bypassing numerical integration
            entirely -- useful for repeated evaluation inside
            optimization/inverse-design loops).
        If None, assumes monodisperse. Default None

    check_inputs : bool (optional)
        If True, check mie scattering inputs. Default True

    Returns:
    --------
    S_q : float
        Structure factor evaluated at q.

    Raises:
    -------
    ValueError
        If inputs are inconsistent or invalid.
    """
    if isinstance(theta, float): theta = _np.array([theta])

    orig_size_dist = size_dist
    is_dist_obj = size_dist is not None and not isinstance(size_dist, _np.ndarray)

    if check_inputs:
        if is_dist_obj:
            # Analytic path never touches D/size_dist as arrays; only
            # wavelength/Nh need validating.
            wavelength, Nh, _, _, _ = _check_mie_inputs(wavelength, Nh)
        else:
            wavelength, Nh, _, D, size_dist = _check_mie_inputs(wavelength, Nh, D=D,
                                                         size_dist=size_dist)

    # compute scattering vector (q = 2k0*sin(theta/2))
    k0 = 2*_np.pi*Nh.real/wavelength
    q = _np.outer(2*k0, _np.sin(theta/2))

    if is_dist_obj:
        S_q = orig_size_dist._closed_form_S(fv, q).T
    elif size_dist is None:
        S_q = _mono_percus_yevick(fv, q, D[-1]).T
    else:
        S_q = _poly_percus_yevick(fv, q, D[-1], size_dist).T

    return S_q


# ---------------------------------------------------------------------------
# Closed-form radius distributions (Botet, Kwok & Cabane 2020)
# ---------------------------------------------------------------------------
# All distributions are NUMBER distributions n(D) over diameter D (Botet's
# n(a), a = D/2), matching the tabulated size_dist convention. Each exposes:
#   .pdf(D)                -- unnormalized number density on an arbitrary D grid
#   .discretize(...)       -- auto-generate a (D_grid, weights) tabulation sized
#                              to avoid tail-truncation / oscillation-resolution
#                              error, for callers needing concrete per-bin
#                              diameters (the Mie-coefficient loops in
#                              phase_scatt_ensemble / cross_section_ensemble)
#   ._closed_form_S(fv, q) -- analytic S(q), bypassing trapz entirely
#
# Scope boundary: these describe a single, homogeneous-sphere size ensemble.
# They are not supported for multilayer D (a list of per-layer arrays).

_X_EPS = 1e-8  # guards the closed-form kernels' 1/x, 1/x**2, 1/x**3 terms at q=0


@_dataclass(frozen=True)
class SchulzDistribution:
    """Schulz-distributed diameters (Botet et al. 2020, Eqs. 34-47)."""
    D_mean: float
    s: float
    kind: str = _field(default='schulz', init=False)

    def __post_init__(self):
        if self.D_mean <= 0:
            raise ValueError("D_mean must be positive.")
        if self.s <= 0:
            raise ValueError("Schulz shape parameter s must be positive.")

    def pdf(self, D):
        """Unnormalized number density n(D) ~ D**(s-1) * exp(-s*D/D_mean) (Eq. 34).

        Evaluated in log-space and re-based at the distribution mode before
        exponentiating: for narrow distributions (large s -- e.g. s~1000 for a
        3% PDI), D**(s-1) and exp(-s*D/D_mean) individually underflow to 0.0
        in float64 (D < 1 raised to a ~1000 power), which previously made
        pdf(D) identically 0 everywhere and corrupted discretize()'s
        normalized weights into NaN. Re-basing at the mode keeps the peak
        pdf value at O(1) regardless of s; this is still an unnormalized
        density (callers normalize the weights themselves), so rescaling by
        a constant does not change any downstream result.
        """
        D = _np.asarray(D, dtype=float)
        s = self.s
        D_mode = self.D_mean * (s - 1) / s if s > 1 else self.D_mean
        log_pdf = (s - 1) * _np.log(D) - s * D / self.D_mean
        log_pdf -= (s - 1) * _np.log(D_mode) - s * D_mode / self.D_mean
        return _np.exp(log_pdf)

    def discretize(self, n_bins=200, n_std=6.0):
        """Auto-generate a (D_grid, weights) tabulation for the Mie-coefficient loop."""
        std_D = self.D_mean / _np.sqrt(self.s)
        lo = max(1e-9, self.D_mean - n_std * std_D)
        hi = self.D_mean + n_std * std_D
        D_grid = _np.linspace(lo, hi, n_bins)
        w = self.pdf(D_grid)
        return D_grid, w / w.sum()

    def _closed_form_S(self, fv, q):
        """Return S(q) via Botet Eqs. 39-47, 5-6 (bypasses trapz)."""
        psi = 3.0 * fv / (1.0 - fv)
        s = self.s
        a_mean = self.D_mean / 2.0
        x = 2.0 * q * a_mean / s  # Eq. 36
        x = _np.where(_np.abs(x) < _X_EPS, _X_EPS, x)
        theta = (s + 2) * _np.arctan(x)
        R = _np.cos(theta) / (1 + x**2)**(1 + s/2)  # Eq. 37
        I = _np.sin(theta) / (1 + x**2)**(1 + s/2)  # Eq. 38
        g1 = psi / ((s + 2) * x)                             # Eq. 39
        g2 = 2*psi / ((s + 1)*(s + 2)*x**2)                   # Eq. 40
        g3 = 4*psi / (s*(s + 1)*(s + 2)*x**3)                 # Eq. 41
        b = (g3 - (s+4)/s*g1)*I - 2*(s+2)/s*g2*R              # Eq. 42
        c = g1 + g3 - (g3 - (s+4)/s*g1)*R - 2*(s+2)/s*g2*I    # Eq. 43
        d = 1 + g1*I                                          # Eq. 44
        e = g1 - g1*R                                         # Eq. 45
        f = g2 - g2*R - (s+3)/(s+1)*g1*I                      # Eq. 46
        g = g1 - g2*I + (s+3)/(s+1)*g1*R                      # Eq. 47
        return _py_assemble_Sq(b, c, d, e, f, g)


def schulz(D_mean, s):
    """
    Build a Schulz-distributed diameter ensemble for use as `size_dist` in
    structure_factor_PY / phase_scatt_ensemble / cross_section_ensemble.

    Parameters
    ----------
    D_mean : float
        Number-mean diameter.
    s : float
        Schulz shape parameter (s > 0). Relative variance p = 1/s
        (Botet et al. 2020, Eq. 35).

    Returns
    -------
    SchulzDistribution
    """
    return SchulzDistribution(D_mean=float(D_mean), s=float(s))


@_dataclass(frozen=True)
class TruncatedNormalDistribution:
    """Truncated-normal diameters (Botet et al. 2020, Eqs. 48-60)."""
    D_mean: float
    p: float
    kind: str = _field(default='truncnorm', init=False)

    def __post_init__(self):
        if self.D_mean <= 0:
            raise ValueError("D_mean must be positive.")
        if self.p <= 0:
            raise ValueError("Polydispersity p must be positive.")
        if self.p >= 0.18:
            _warnings.warn(
                f"truncated_normal polydispersity p={self.p:.3f} >= 0.18: "
                "Botet et al. (2020) validated this closed form only for "
                "p < 0.18; results beyond that are extrapolated.",
                UserWarning,
            )

    def pdf(self, D):
        """Unnormalized number density n(D) ~ exp(-(D-D_mean)**2/(2*p*D_mean**2)) (Eq. 48)."""
        D = _np.asarray(D, dtype=float)
        return _np.exp(-(D - self.D_mean)**2 / (2*self.p*self.D_mean**2))

    def discretize(self, n_bins=200, n_std=6.0):
        """Auto-generate a (D_grid, weights) tabulation for the Mie-coefficient loop."""
        std_D = self.D_mean * _np.sqrt(self.p)
        lo = max(1e-9, self.D_mean - n_std * std_D)
        hi = self.D_mean + n_std * std_D
        D_grid = _np.linspace(lo, hi, n_bins)
        w = self.pdf(D_grid)
        return D_grid, w / w.sum()

    def _closed_form_S(self, fv, q):
        """Return S(q) via Botet Eqs. 50-60, 5-6 (bypasses trapz). Valid for p < 0.18."""
        psi = 3.0*fv/(1.0 - fv)
        p = self.p
        x = q * self.D_mean  # Eq. 49: x = 2 q <a>, <a> = D_mean/2
        x = _np.where(_np.abs(x) < _X_EPS, _X_EPS, x)
        R = 3*fv/((1+3*p)*(1-fv)*x**3) * _np.exp(-p*x**2/2) * _np.cos(x)   # Eq. 50
        I = 3*fv/((1+3*p)*(1-fv)*x**3) * _np.exp(-p*x**2/2) * _np.sin(x)   # Eq. 51
        g1 = psi*(1+p)/((1+3*p)*x)          # Eq. 52
        g2 = 2*psi/((1+3*p)*x**2)           # Eq. 53
        g3 = 4*psi/((1+3*p)*x**3)           # Eq. 54
        b = ((2+p*x**2)**2 - x**2*(1+p))*I - 2*x*(2+p*x**2)*R                       # Eq. 55
        c = g1+g3 - 2*x*(2+p*x**2)*I - ((2+p*x**2)**2 - x**2*(1+p))*R              # Eq. 56
        d = 1 + x**2*(1+p-p**2*x**2)*I + 2*p*x**3*R                                # Eq. 57
        e = g1 + 2*p*x**3*I - (1+p-p**2*x**2)*x**2*R                               # Eq. 58
        f = g2 - 2*x*(1+p*x**2)*R - x**2*(1-p-p**2*x**2)*I                         # Eq. 59
        g = g1 + x**2*(1-p-p**2*x**2)*R - 2*x*(1+p*x**2)*I                         # Eq. 60
        return _py_assemble_Sq(b, c, d, e, f, g)


def truncated_normal(D_mean, p):
    """
    Build a truncated-normal diameter ensemble for use as `size_dist` in
    structure_factor_PY / phase_scatt_ensemble / cross_section_ensemble.

    Botet et al. (2020) validated this closed form for polydispersity
    p < 0.18 (18%); p >= 0.18 emits a warning and is extrapolated.

    Parameters
    ----------
    D_mean : float
        Number-mean diameter.
    p : float
        Relative variance of the diameter distribution.

    Returns
    -------
    TruncatedNormalDistribution
    """
    return TruncatedNormalDistribution(D_mean=float(D_mean), p=float(p))


@_dataclass(frozen=True)
class InverseGaussianDistribution:
    """Inverse-Gaussian diameters (Botet et al. 2020, Eqs. 61-68)."""
    D_mean: float
    p: float
    kind: str = _field(default='invgauss', init=False)

    def __post_init__(self):
        if self.D_mean <= 0:
            raise ValueError("D_mean must be positive.")
        if self.p <= 0:
            raise ValueError("Polydispersity p must be positive.")

    def pdf(self, D):
        """Unnormalized number density n(D) ~ D**-1.5 * exp(-(D-D_mean)**2/(2*p*D_mean*D)) (Eq. 61)."""
        D = _np.asarray(D, dtype=float)
        return D**(-1.5) * _np.exp(-(D - self.D_mean)**2 / (2*self.p*self.D_mean*D))

    def discretize(self, n_bins=200, n_std=8.0):
        """Auto-generate a (D_grid, weights) tabulation for the Mie-coefficient loop."""
        std_D = self.D_mean * _np.sqrt(self.p)
        lo = max(1e-9, self.D_mean - n_std * std_D)
        hi = self.D_mean + n_std * std_D
        D_grid = _np.linspace(lo, hi, n_bins)
        w = self.pdf(D_grid)
        return D_grid, w / w.sum()

    def _closed_form_S(self, fv, q):
        """Return S(q) via the complex mu_1,mu_2,mu_3 form (Botet Eqs. 64-68, 92-100),
        converted to the shared real b,c,d,e,f,g assembly via the paper's stated
        identity f_11=b+ic, f_22=d+ie, f_12=f+ig."""
        psi = 3.0*fv/(1.0-fv)
        p = self.p
        a_mean = self.D_mean/2.0
        x = q*a_mean  # Eq. 61 note: x = q<a>
        x = _np.where(_np.abs(x) < _X_EPS, _X_EPS, x)
        one_m4ixp = 1 - 4j*x*p
        sqrt_term = one_m4ixp**0.5
        mu0 = (1.0/one_m4ixp) * _np.exp((1 - sqrt_term)/p)                         # Eq. 62
        mu1 = (1j/2)*(1 + (1+p)*x**2                                                # Eq. 64
                       - mu0*(1 - 4j*x*p - x**2 - 1j*x*(2-9j*x*p)/sqrt_term))
        mu2 = (1j*x**2/2)*(1+p - mu0*(1 + p/sqrt_term))                            # Eq. 65
        mu3 = (x/2)*(1 + 1j*x*(1+p) + mu0*(1j*x - (1-5j*x*p)/sqrt_term))           # Eq. 66
        nu3 = (1+3*p+3*p**2)*x**3                                                   # Eq. 68
        # Note: empirically (cross-checked against the trapz kernel), the
        # "+1" that appears in the general f_11/f_22 definitions elsewhere
        # in the paper does not apply to f_11 in this inverse-Gaussian
        # closed form -- only f_22 carries it here.
        f11 = psi*mu1/nu3       # Eq. 97
        f22 = 1 + psi*mu2/nu3   # Eq. 98
        f12 = psi*mu3/nu3       # Eq. 99
        b, c = _np.real(f11), _np.imag(f11)
        d, e = _np.real(f22), _np.imag(f22)
        f, g = _np.real(f12), _np.imag(f12)
        return _py_assemble_Sq(b, c, d, e, f, g)


def inverse_gaussian(D_mean, p):
    """
    Build an inverse-Gaussian diameter ensemble for use as `size_dist` in
    structure_factor_PY / phase_scatt_ensemble / cross_section_ensemble.

    Parameters
    ----------
    D_mean : float
        Number-mean diameter.
    p : float
        Relative variance of the diameter distribution.

    Returns
    -------
    InverseGaussianDistribution
    """
    return InverseGaussianDistribution(D_mean=float(D_mean), p=float(p))


@_dataclass(frozen=True)
class ExponentialDistribution:
    """Exponentially-distributed diameters (Botet et al. 2020, Eqs. 17-27)."""
    D_mean: float
    kind: str = _field(default='exponential', init=False)

    def __post_init__(self):
        if self.D_mean <= 0:
            raise ValueError("D_mean must be positive.")

    def pdf(self, D):
        """Unnormalized number density n(D) ~ exp(-D/D_mean) (Eq. 17)."""
        D = _np.asarray(D, dtype=float)
        return _np.exp(-D / self.D_mean)

    def discretize(self, n_bins=200, n_tail=12.0):
        """Auto-generate a (D_grid, weights) tabulation for the Mie-coefficient loop."""
        lo = max(1e-9, self.D_mean * 1e-3)
        hi = self.D_mean * n_tail
        D_grid = _np.linspace(lo, hi, n_bins)
        w = self.pdf(D_grid)
        return D_grid, w / w.sum()

    def _closed_form_S(self, fv, q):
        """Return S(q) via Botet Eqs. 19-24, 5-6 (bypasses trapz)."""
        x = q * self.D_mean  # Eq. 18: x = 2 q <a>, <a> = D_mean/2
        x = _np.where(_np.abs(x) < _X_EPS, _X_EPS, x)
        ratio = fv / (1.0 - fv)
        b = ratio * (1 + 5*x**2) / (1 + x**2)**3                    # Eq. 19
        c = ratio * x**3*(5 + x**2) / (1 + x**2)**3                  # Eq. 20
        d = 1 + ratio * (3 - x**2) / (1 + x**2)**3                   # Eq. 21
        e = ratio * x*(6 + 3*x**2 + x**4) / (1 + x**2)**3            # Eq. 22
        f = ratio * x**2*(5 + x**2) / (1 + x**2)**3                  # Eq. 23
        g = ratio * x*(-2 + 3*x**2 + x**4) / (1 + x**2)**3           # Eq. 24
        return _py_assemble_Sq(b, c, d, e, f, g)


def exponential(D_mean):
    """
    Build an exponentially-distributed diameter ensemble for use as
    `size_dist` in structure_factor_PY / phase_scatt_ensemble /
    cross_section_ensemble.

    Parameters
    ----------
    D_mean : float
        Number-mean diameter.

    Returns
    -------
    ExponentialDistribution
    """
    return ExponentialDistribution(D_mean=float(D_mean))
