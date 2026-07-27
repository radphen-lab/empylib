# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:38:11 2021

@author: PanxoPanza
"""
import numpy as _np
from numpy import pi as _pi, exp as _exp, conj as _conj, imag as _imag, real as _real
from numpy.polynomial.legendre import leggauss as _leggauss
from scipy.special import hankel1e as _hankel1e, jve as _jve, eval_legendre as _eval_legendre
from scipy.integrate import simpson as _simpson
from .nklib import emt_brugg as _emt_brugg, emt_multilayer_sphere as _emt_multilayer_sphere
from .utils import (
    _as_1d_array,
    _check_mie_inputs,
    _check_theta,
    _hide_signature,
    _effective_host,
)
# structure_factor_PY (and its underlying _mono_percus_yevick /
# _poly_percus_yevick kernels, plus the closed-form distribution builders:
# schulz, truncated_normal, inverse_gaussian, exponential) live in
# dense_spheres.py. Re-exported here so `mie.structure_factor_PY(...)`
# keeps working unchanged.
from .dense_spheres import structure_factor_PY
import pandas as _pd
from typing import Union as _Union, List as _List, Optional as _Optional, Tuple as _Tuple

__all__ = (
    'scatter_efficiency',
    'scatter_coefficients',
    'scatter_amplitude',
    'scatter_stokes',
    'phase_scatt_HG',
    'scatter_from_phase_function',
    'structure_factor_PY',
    'phase_scatt_ensemble',
    'phase_function_moments',
    'cross_section_ensemble',
)

def _safe_complex_divide(num, den, *, floor=1e-300):
    """Complex division with a tiny denominator floor for recurrence pivots."""
    num = _np.asarray(num, dtype=_np.complex128)
    den = _np.asarray(den, dtype=_np.complex128)
    den_safe = _np.where(_np.abs(den) < floor, den + floor, den)
    return num / den_safe

def _default_nmax(y):
    """Wiscombe/Johnson (1996) truncation order for size parameter(s) y."""
    y_max = _np.max(_np.abs(y))
    return int(_np.round(y_max + 4 * y_max ** (1 / 3) + 2))

def _host_size_parameter(wavelength, Nh, R):
    """Return the complex host size parameter k_h R."""
    wavelength = _np.asarray(wavelength, dtype=float)
    Nh = _np.asarray(Nh, dtype=_np.complex128)
    if _np.any(_imag(Nh) < -1e-12):
        raise ValueError("Complex host refractive index with negative imaginary part is not supported.")
    kh = 2 * _pi * Nh / wavelength
    return _np.outer(kh, _np.asarray(R, dtype=float))

def _riccati_psi_scaled(z, nmax):
    """Return psi_n(z) scaled by exp(-Im(z)) for Im(z) >= 0."""
    z = _np.asarray(z, dtype=_np.complex128)
    n = _np.arange(1, nmax + 1)
    nu = n + 0.5
    return _np.sqrt(0.5 * _pi * z[..., None]) * _jve(nu.reshape((1,) * z.ndim + (-1,)), z[..., None])

def _riccati_xi_over_psi(z, nmax):
    """Stable inverse Riccati ratio rho_n = xi_n(z) / psi_n(z), via Bessel functions.

    Exact but expensive: for complex ``z`` SciPy routes ``jve``/``hankel1e``
    through its Fortran backend for complex arguments, which costs ~14 us per
    element and does not vectorize.  ``_log_RicattiBessel`` therefore obtains
    rho from a recurrence instead (see ``_rho_from_DnGn``) and calls this only
    for the few rows where that recurrence's seed is ill-conditioned.  Kept as
    the reference implementation the recurrence is validated against.
    """
    z = _np.asarray(z, dtype=_np.complex128)
    n = _np.arange(1, nmax + 1)
    nu = n + 0.5
    z_expanded = z[..., None]
    nu_expanded = nu.reshape((1,) * z.ndim + (-1,))

    with _np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        j_scaled = _jve(nu_expanded, z_expanded)
        h_scaled = _hankel1e(nu_expanded, z_expanded)
        scale = _np.exp(1j * _real(z_expanded) - 2.0 * _imag(z_expanded))
        rho = h_scaled * scale / j_scaled

    return rho

# Below this magnitude of xi_0/psi_0's denominator the seed of the rho
# recurrence loses its significant digits; such rows fall back to the exact
# Bessel evaluation.  On a 161-wavelength sweep this typically selects <10 rows.
_RHO_SEED_FLOOR = 1e-6

def _rho_from_DnGn(x, Dnx, Gnx, nmax):
    """rho_n = xi_n(x)/psi_n(x) by upward recurrence on the Wu & Wang ratios.

    Inverting the psi/xi recurrence of Wu & Wang (1991) gives

        rho_n = rho_{n-1} * (D_n + (n+1)/x) / (G_n + (n+1)/x)

    seeded with rho_0 = xi_0/psi_0 = 2/(1 - exp(-2ix)).  Recurring rho (rather
    than R = psi/xi) walks the *growing* solution, so it stays stable where the
    R recurrence loses accuracy for absorbing hosts.

    The seed is written as 2*e2/(e2 - 1) with e2 = exp(2ix): for Im(x) >= 0
    that has |e2| <= 1 and therefore cannot overflow, unlike the algebraically
    equivalent 2/(1 - exp(-2ix)).  It is ill-conditioned only where e2 -> 1,
    i.e. real x near a multiple of pi, where psi_0 = sin(x) -> 0; those rows,
    and any with Im(x) < 0 (outside this seed's assumption), are recomputed
    with _riccati_xi_over_psi.

    Parameters
    ----------
    x : (n_wavelengths, n_shells) complex ndarray
    Dnx, Gnx : (n_wavelengths, n_shells, >= nmax) complex ndarray
        Logarithmic derivatives already computed by ``_log_RicattiBessel``.
    """
    with _np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        e2 = _np.exp(2j * x)
        seed_den = e2 - 1.0
        rho_prev = 2.0 * e2 / seed_den

        rho = _np.zeros(x.shape + (nmax,), dtype=_np.complex128)
        for i in range(nmax):
            shift = (i + 1) / x
            rho_prev = rho_prev * (Dnx[:, :, i] + shift) / (Gnx[:, :, i] + shift)
            rho[:, :, i] = rho_prev

    # Exact fallback where the seed carries too few significant digits.
    bad = (_np.abs(seed_den) < _RHO_SEED_FLOOR) | (_imag(x) < 0.0) | ~_np.isfinite(rho_prev)
    if _np.any(bad):
        rho[bad, :] = _riccati_xi_over_psi(x[bad], nmax)

    return rho

def _log_RicattiBessel(x,nmax,nmx):
    '''
    Computes the logarithmic derivatives of Ricatti-Bessel functions,
        Dn(x) = psi_n'(x) / psi_n(x),
        Gn(x) = chi_n'(x) / chi_n(x), and
        rho_n(x) = xi_n(x) / psi_n(x);
    using the method by Wu & Wang Radio Sci. 26, 1393–1401 (1991).

    Parameters
    ----------
    x : 1D numpy array
        size parameter for each shell
    nmax : int
        number of mie coefficients
    nmx : int
        extended value of nmax for downward recursion (Wu & Wang, 1991)

    Returns
    -------
    1D numpy array
        Dn(x)
    1D numpy array
        Gn(x)
    1D numpy array
        rho_n(x)
    '''
    
    # Internal convention: (n_wavelengths, n_shells). Keep scalar/1D compatibility
    # by promoting to 2D and squeezing back at return.
    x = _np.asarray(x, dtype=_np.complex128)
    # If input was scalar/1D, return scalar/1D outputs to preserve old API behavior.
    squeeze_out = False
    if x.ndim == 0:
        x = x.reshape(1, 1)
        squeeze_out = True
    elif x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze_out = True
    elif x.ndim != 2:
        raise ValueError("x must be scalar, 1D, or 2D.")

    n = _np.arange(nmax)

    # Get Dn(x) by downwards recurrence
    Dnx = _np.zeros((x.shape[0], x.shape[1], nmx), dtype=_np.complex128)
    for i in reversed(range(1, nmx)):
        Dnx[:, :, i - 1] = (i + 1) / x - 1 / (Dnx[:, :, i] + (i + 1) / x)

    # Get Gn(x) by upwards recurrence
    Gnx = _np.zeros((x.shape[0], x.shape[1], nmx), dtype=_np.complex128)
    G0x = 1j * _np.ones_like(x)
    i = 0
    Gnx[:, :, i] = 1 / ((i + 1) / x - G0x) - (i + 1) / x
    for i in range(1, nmx):
        Gnx[:, :, i] = 1 / ((i + 1) / x - Gnx[:, :, i - 1]) - (i + 1) / x

    # Stable inverse ratio rho = xi / psi.  Computing R = psi / xi directly
    # overflows for absorbing hosts because psi grows as exp(Im(x)); recurring
    # rho instead reuses the Dn/Gn arrays just built, so no Bessel evaluation
    # is needed except for the few ill-conditioned rows _rho_from_DnGn detects.
    rhonx = _rho_from_DnGn(x, Dnx, Gnx, nmax)

    Dn = Dnx[:, :, n]
    Gn = Gnx[:, :, n]
    if squeeze_out:
        return Dn[0], Gn[0], rhonx[0]
    return Dn, Gn, rhonx

def _recursive_ab(m, n, Dn, Gn, rho, Dn1, Gn1, rho1):
    """
    Compute normalized multilayer Mie coefficients using Johnson-style layer
    recursion, vectorized over wavelength.

    Notes
    -----
    - Internal working shape is ``(n_wavelengths, n_layers, n_orders)``.
    - The returned values are ``alpha = an / R`` and ``beta = bn / R`` at the
      outer host boundary, where ``R = psi/xi``.  Keeping this normalization
      avoids overflow for complex absorbing host size parameters.
    - For single-wavelength/single-particle calls, inputs may arrive as 2D;
      they are temporarily promoted to 3D and squeezed back on return.
    """
    m = _np.asarray(m, dtype=_np.complex128)
    Dn = _np.asarray(Dn, dtype=_np.complex128)
    Gn = _np.asarray(Gn, dtype=_np.complex128)
    rho = _np.asarray(rho, dtype=_np.complex128)
    Dn1 = _np.asarray(Dn1, dtype=_np.complex128)
    Gn1 = _np.asarray(Gn1, dtype=_np.complex128)
    rho1 = _np.asarray(rho1, dtype=_np.complex128)

    # If we promoted scalar-style inputs to batched shape, restore on return.
    squeeze_out = False
    if Dn.ndim == 2:
        # Promote to 3D so one vectorized implementation covers both cases.
        Dn = Dn.reshape(1, Dn.shape[0], Dn.shape[1])
        Gn = Gn.reshape(1, Gn.shape[0], Gn.shape[1])
        rho = rho.reshape(1, rho.shape[0], rho.shape[1])
        Dn1 = Dn1.reshape(1, Dn1.shape[0], Dn1.shape[1])
        Gn1 = Gn1.reshape(1, Gn1.shape[0], Gn1.shape[1])
        rho1 = rho1.reshape(1, rho1.shape[0], rho1.shape[1])
        m = m.reshape(1, -1)
        squeeze_out = True

    n_layers = Dn.shape[1]
    # Start from the core boundary condition before shell-by-shell updates.
    alpha = _np.zeros((Dn.shape[0], n), dtype=_np.complex128)
    beta = _np.zeros((Dn.shape[0], n), dtype=_np.complex128)

    # Layer-by-layer Johnson recursion, vectorized over wavelength.
    for i in range(1, n_layers + 1):
        # Relative index jump at interface i-1 -> i.
        ratio = (m[:, i] / m[:, i - 1]).reshape(-1, 1)

        # Auxiliary interface terms in Johnson's recursion.
        Un = _safe_complex_divide(Dn[:, i - 1, :] - alpha * Gn[:, i - 1, :], 1.0 - alpha)
        Vn = _safe_complex_divide(Dn[:, i - 1, :] - beta * Gn[:, i - 1, :], 1.0 - beta)

        alpha_next = _safe_complex_divide(
            ratio * Un - Dn1[:, i - 1, :],
            ratio * Un - Gn1[:, i - 1, :],
        )
        beta_next = _safe_complex_divide(
            Vn - ratio * Dn1[:, i - 1, :],
            Vn - ratio * Gn1[:, i - 1, :],
        )

        if i < n_layers:
            # Move the normalized coefficient from the inner boundary of the
            # next layer to that layer's outer boundary.
            scale = _safe_complex_divide(rho[:, i, :], rho1[:, i - 1, :])
            alpha = alpha_next * scale
            beta = beta_next * scale
        else:
            alpha = alpha_next
            beta = beta_next

    if squeeze_out:
        return alpha[0], beta[0]
    return alpha, beta
        
def _coeffs_finite(an, bn):
    """True if an/bn are non-None, fully finite, and safely summable.

    "Safely summable" means below sqrt(float max)/100, the threshold used
    throughout this module to guard against overflow in an*conj(bn)-style
    products.
    """
    if an is None or bn is None:
        return False
    raw_limit = _np.sqrt(_np.finfo(float).max) / 100.0
    finite = _np.all(_np.isfinite(an)) and _np.all(_np.isfinite(bn))
    return bool(
        finite
        and _np.nanmax(_np.abs(an)) < raw_limit
        and _np.nanmax(_np.abs(bn)) < raw_limit
    )

def _clip_small_negative(q, *extra_refs):
    """Snap tiny negative numerical noise in q to 0.

    tol = 1e-10 + 1e-8 * max(|q|, |extra_refs...|). Values below -tol are
    left untouched (treated as genuinely negative, not noise).
    """
    q = _np.asarray(q)
    tol = 1e-10 + 1e-8 * _np.maximum.reduce([_np.abs(q)] + [_np.abs(r) for r in extra_refs])
    return _np.where((q < 0) & (_np.abs(q) <= tol), 0.0, q)

def _reconstruct_raw_coefficients(alpha, beta, rho_outer, *, raise_on_overflow=False):
    with _np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        an = alpha * _safe_complex_divide(1.0, rho_outer)
        bn = beta * _safe_complex_divide(1.0, rho_outer)

    safely_summable = _coeffs_finite(an, bn)
    if raise_on_overflow and not safely_summable:
        raise FloatingPointError(
            "Raw Mie coefficients overflow for this complex host size parameter; "
            "use efficiency/cross-section APIs, which use the stable normalized coefficients."
        )
    return an, bn

def _get_coated_state(m, x, nmax=None, *, reconstruct_raw=True, raise_on_raw_overflow=False):
    """Compute coated-sphere coefficient state with overflow-safe normalization."""
    x = _np.asarray(x, dtype=_np.complex128)
    m = _np.asarray(m, dtype=_np.complex128)
    squeeze_out = False

    if x.ndim == 1:
        x = x.reshape(1, -1)
        m = m.reshape(1, -1)
        squeeze_out = True
    elif x.ndim != 2:
        raise ValueError("x must be 1D or 2D.")

    if m.shape != x.shape:
        raise ValueError("m and x must have the same shape.")

    ka = x[:, -1]
    if nmax is None:
        nmax = _default_nmax(ka)

    mix = m * x
    mi1 = _np.concatenate((m, _np.ones((m.shape[0], 1), dtype=_np.complex128)), axis=1)
    mi1x = mi1[:, 1:] * x
    nmx = int(_np.round(max(nmax, _np.max(abs(mix))) + 16))

    Dn, Gn, rho = _log_RicattiBessel(mix, nmax, nmx)
    Dn1, Gn1, rho1 = _log_RicattiBessel(mi1x, nmax, nmx)
    alpha, beta = _recursive_ab(mi1, nmax, Dn, Gn, rho, Dn1, Gn1, rho1)

    rho_outer = rho1[:, -1, :]
    Dy = Dn1[:, -1, :]
    Gy = Gn1[:, -1, :]

    an = bn = None
    if reconstruct_raw:
        an, bn = _reconstruct_raw_coefficients(
            alpha, beta, rho_outer, raise_on_overflow=raise_on_raw_overflow
        )

    # psi_n(ka) is deliberately NOT computed here.  Its only consumer is the
    # absorbing-host Johnson Qext/Qsca formulas in _cross_section_at_lam, which
    # are discarded for a lossless host, and it costs a full scipy Bessel
    # evaluation over the (n_wavelengths, nmax) grid.  That caller computes it
    # on demand -- see _psi_scaled_for.
    state = {
        "alpha": alpha,
        "beta": beta,
        "an": an,
        "bn": bn,
        "psi_scaled": None,
        "Dy": Dy,
        "Gy": Gy,
        "rho_outer": rho_outer,
        "nmax": nmax,
        "ka": ka,
    }

    if squeeze_out:
        for key in ("alpha", "beta", "an", "bn", "psi_scaled", "Dy", "Gy", "rho_outer", "ka"):
            if state[key] is not None:
                state[key] = state[key][0]
    return state

def _cross_section_at_lam(m,x,nmax = None, *, return_state: bool = False):
    '''
    Compute mie scattering parameters for a given lambda
    The absorption, scattering, extinction and asymmetry parameter are 
    computed with the formulas for absorbing medium reported in 
    
    - Johnson, B. R. Light scattering by a multilayer sphere (1996). App. Opt., 
        35(18), 3286.
    
    - Wu, Z. S.; Wang, Y. P. (1991). Electromagnetic scattering for 
        multilayered sphere: Recursive algorithms. Science, 26(6), 1393–1401.

    Parameters
    ----------
    m : 1D numpy array
        normalized refractive index of shell layers
    x : 1D numpy array
        size paramter for each shell layer
    nmax : int, optional
        number of mie coefficients. The default is -1.

    Returns
    -------
    Qext : float
        Extinction efficiency.
    Qsca : float
        Scattering efficiency.
    Asym : float (-1, 1)
        Asymmetry parameter.
    Qb : float
        Backward scattering effiency.
    Qf : float
        Forward scatttering efficiency.
    nmax : int
        number of mie coefficients.
    an : 1D numpy array (size nmax)
        mie coefficient for M function.
    bn : 1D numpy array (size nmax)
        mie coefficient for N function.
    '''
    m = _np.asarray(m, dtype=_np.complex128)
    x = _np.asarray(x)
    # Keep old scalar API while computing internally in batched 2D form.
    squeeze_out = False

    if x.ndim == 1:
        x = x.reshape(1, -1)
        m = m.reshape(1, -1)
        squeeze_out = True
    elif x.ndim != 2:
        raise ValueError("x must be 1D or 2D.")

    if m.shape != x.shape:
        raise ValueError("m and x must have the same shape.")

    # determine nmax
    y = x[:, -1] # size parameter of outer layer

    if nmax is None :
        # define nmax according to B.R Johnson (1996)
        nmax = _default_nmax(y)

    #------------------------------------------------------------------
    # Single batched coefficient call for all wavelengths.
    #------------------------------------------------------------------
    state = _get_coated_state(m, x, nmax, reconstruct_raw=True, raise_on_raw_overflow=False)
    alpha = state["alpha"]
    beta = state["beta"]
    an = state["an"]
    bn = state["bn"]
    Dy = state["Dy"]
    Gy = state["Gy"]
    nmax = state["nmax"]

    # Matched-index absorbing-host cases are especially sensitive to
    # cancellation in the Johnson formulas below.  If the full Mie series is
    # already numerically extinguished, snap the coefficients to zero so the
    # derived efficiencies follow that same limit consistently.
    coeff_floor = 1e-10
    coeff_mask = _np.maximum(_np.max(_np.abs(alpha), axis=1), _np.max(_np.abs(beta), axis=1)) < coeff_floor
    if _np.any(coeff_mask):
        alpha = alpha.copy()
        beta = beta.copy()
        alpha[coeff_mask, :] = 0.0
        beta[coeff_mask, :] = 0.0

    # arranging pre-computing constants
    n = _np.arange(1, nmax + 1, dtype=float).reshape(1, -1)

    #------------------------------------------------------------------
    # Lossless-host efficiencies from the raw an/bn series.
    #
    # Evaluated FIRST so we know which rows it covers.  Any row it covers
    # never reads the Johnson absorbing-host formulas below, which are the
    # sole consumer of psi_n(y) -- so for an entirely lossless host the
    # scipy Bessel evaluation of psi is skipped altogether.
    #------------------------------------------------------------------
    Qext = _np.zeros(_real(y).shape, dtype=float)
    Qsca = _np.zeros(_real(y).shape, dtype=float)
    covered = _np.zeros(_real(y).shape, dtype=bool)

    coeffs_finite = _coeffs_finite(an, bn)
    near_lossless_host = _np.abs(_imag(y)) <= 1e-7 * _np.maximum(1.0, _np.abs(_real(y)))
    if coeffs_finite and _np.any(near_lossless_host):
        y_real = _real(y[near_lossless_host]).reshape(-1, 1)
        an_lossless = an[near_lossless_host, :]
        bn_lossless = bn[near_lossless_host, :]
        weights = 2*n + 1
        qext_lossless = _real(
            2.0 / y_real[:, 0]**2 * _np.sum(weights * _real(an_lossless + bn_lossless), axis=1)
        )
        qsca_lossless = _real(
            2.0 / y_real[:, 0]**2
            * _np.sum(weights * (_np.abs(an_lossless)**2 + _np.abs(bn_lossless)**2), axis=1)
        )
        q_tol_lossless = 1e-10 + 1e-8 * _np.maximum(_np.abs(qext_lossless), _np.abs(qsca_lossless))
        valid_lossless = qext_lossless >= (qsca_lossless - q_tol_lossless)
        lossless_rows = _np.flatnonzero(near_lossless_host)
        if _np.any(valid_lossless):
            rows = lossless_rows[valid_lossless]
            Qext[rows] = qext_lossless[valid_lossless]
            Qsca[rows] = qsca_lossless[valid_lossless]
            covered[rows] = True

    #------------------------------------------------------------------
    # Absorbing-host efficiencies, Johnson (1996).  The correction factor is
    # combined with |psi|^2 in scaled form so complex host size parameters
    # with large Im(y) do not overflow.  Skipped entirely when the lossless
    # branch above already covers every row.
    #------------------------------------------------------------------
    if not _np.all(covered):
        imy = 2*_imag(y)
        ft_scaled = _np.full_like(_real(y), 2.0, dtype=float)
        mask_abs_host = _imag(y) > 1E-8
        if _np.any(mask_abs_host):
            imy_sel = imy[mask_abs_host]
            ft_scaled[mask_abs_host] = _real(
                imy_sel**2 / (_exp(-imy_sel) + (imy_sel - 1.0))
            )
        py_scaled = _riccati_psi_scaled(y, nmax)
        state["psi_scaled"] = py_scaled
        psi2_scaled = _np.abs(py_scaled) ** 2
        common = psi2_scaled * ft_scaled.reshape(-1, 1)

        # Extinction efficiency
        en = (2*n+1)*_imag(common * (- 2j*_imag(Dy)               \
                           + _conj(alpha)*Dy                      \
                           - _conj(beta)*_conj(Gy)                \
                           + alpha*Gy                             \
                           - beta*_conj(Dy))                      \
                           /y.reshape(-1, 1))
        q = _np.sum(en, axis=1)
        qext_johnson = _real(1/_real(y)*q)

        # Scattering efficiency
        en = (2*n+1)*_imag(common * (+ _np.abs(alpha)**2*Gy       \
                           - _np.abs(beta)**2*_conj(Gy)           \
                           )/y.reshape(-1, 1))
        q = _np.sum(en, axis=1)
        qsca_johnson = _real(1/_real(y)*q)

        Qext = _np.where(covered, Qext, qext_johnson)
        Qsca = _np.where(covered, Qsca, qsca_johnson)
    
    #------------------------------------------------------------------
    # Asymmetry parameter
    #------------------------------------------------------------------
    an_g = an if coeffs_finite else alpha
    bn_g = bn if coeffs_finite else beta

    anp1 = _np.zeros_like(an_g, dtype=_np.complex128)
    bnp1 = _np.zeros_like(bn_g, dtype=_np.complex128)
    anp1[:, :nmax-1] = an_g[:, 1:] # a(n+1) coefficient
    bnp1[:, :nmax-1] = bn_g[:, 1:] # b(n+1) coefficient

    asy1 = n*(n + 2)/(n + 1)*(an_g*_conj(anp1)+ bn_g*_conj(bnp1)) \
         + (2*n + 1)/(n*(n + 1))*_real(an_g*_conj(bn_g))
    
    asy2 = (2*n+1)*(an_g*_conj(an_g) + bn_g*_conj(bn_g))
    asy2_sum = _np.sum(asy2, axis=1)
    with _np.errstate(divide='ignore', invalid='ignore'):
        Asym = _real(
            _np.divide(
                2 * _np.sum(asy1, axis=1),
                asy2_sum,
                out=_np.zeros_like(asy2_sum, dtype=_np.complex128),
                where=~_np.isclose(asy2_sum, 0.0),
            )
        )
    
    #------------------------------------------------------------------
    # Backward scattering (not valid for absorbing host media)
    #------------------------------------------------------------------
    if coeffs_finite:
        f = (2*n+1)*((-1)**n)*(an - bn)
        q = _np.sum(f, axis=1)
        Qb = _real(q*_conj(q)/y**2)
    else:
        Qb = _np.zeros_like(Qext)
    
    #------------------------------------------------------------------
    # Forward scattering (not valid for absorbing host media)
    #------------------------------------------------------------------
    if coeffs_finite:
        f = (2*n+1)*(an + bn)
        q = _np.sum(f, axis=1)
        Qf = _real(q*_conj(q)/y**2)
    else:
        Qf = _np.zeros_like(Qext)
    
    #------------------------------------------------------------------
    # Condition outputs to avoid unphysical results
    #------------------------------------------------------------------
    if _np.any(coeff_mask):
        Qext[coeff_mask] = 0.0
        Qsca[coeff_mask] = 0.0
        Asym[coeff_mask] = 0.0

    Qsca, Qext = _clip_small_negative(Qsca, Qext), _clip_small_negative(Qext, Qsca)
    Asym = _np.clip(Asym, -1, +1)

    if squeeze_out:
        if return_state:
            state["qext"] = Qext
            state["qsca"] = Qsca
            state["gcos"] = Asym
            return Qext[0], Qsca[0], Asym[0], Qb[0], Qf[0], nmax, an[0], bn[0], state
        return Qext[0], Qsca[0], Asym[0], Qb[0], Qf[0], nmax, an[0], bn[0]
    if return_state:
        state["qext"] = Qext
        state["qsca"] = Qsca
        state["gcos"] = Asym
        return Qext, Qsca, Asym, Qb, Qf, nmax, an, bn, state
    return Qext, Qsca, Asym, Qb, Qf, nmax, an, bn

def _normalize_single_particle_inputs(
    wavelength: _Union[float, _np.ndarray],
    Nh: _Union[float, _np.ndarray],
    Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
    D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
    *,
    check_inputs: bool = True,
):
    """
    Normalize the single-particle API to the same scalar/array conventions used
    by ``rad_transfer``.

    When ``check_inputs=True``, this delegates all wavelength/Nh/Np/D
    validation to ``_check_mie_inputs`` and only adds the mono-particle
    specialization (reject polydisperse ``D``, extract the scalar
    ``D_shells`` vector). When ``check_inputs=False``, callers are trusted
    to already provide ``Np`` pre-shaped as ``(n_layers, nlam)`` -- the same
    contract every other ``check_inputs=False`` path in this module relies
    on -- while ``D`` may still arrive in any of its raw forms (scalar,
    1D/2D array, list/tuple).

    Returns
    -------
    wavelength : ndarray, shape (nlam,)
    Nh : ndarray, shape (nlam,)
    Np : ndarray, shape (n_layers, nlam)
    D_layers : list[ndarray]
        One diameter array per shell.
    D_shells : ndarray, shape (n_layers,)
        One scalar diameter per shell. Polydisperse inputs are rejected here.
    """
    if check_inputs:
        wavelength, Nh, Np, D_layers, _ = _check_mie_inputs(wavelength, Nh, Np, D)
    else:
        wavelength = _np.atleast_1d(_np.asarray(wavelength, dtype=float))
        if wavelength.ndim != 1 or wavelength.size == 0:
            raise ValueError("wavelength must be a non-empty 1D array.")

        Nh = _as_1d_array(Nh, "Nh", n_wavelengths=wavelength.size, dtype=complex)
        Np = _np.asarray(Np, dtype=complex)
        if Np.ndim == 1:
            Np = Np.reshape(1, -1)
        if Np.ndim != 2 or Np.shape[1] != wavelength.size:
            raise ValueError("Np must resolve to shape (n_layers, len(wavelength)).")

        if _np.isscalar(D):
            D_layers = [_np.array([float(D)], dtype=float)]
        elif isinstance(D, _np.ndarray):
            if D.ndim == 1:
                D_layers = [_np.asarray(D, dtype=float).ravel()]
            elif D.ndim == 2:
                D_layers = [_np.asarray(row, dtype=float).ravel() for row in D]
            else:
                raise ValueError("D must be a scalar, a 1D/2D array, or a list of layer diameters.")
        elif isinstance(D, (list, tuple)):
            D_layers = [_np.atleast_1d(_np.asarray(layer, dtype=float)).ravel() for layer in D]
        else:
            raise TypeError("D must be a scalar, a 1D/2D array, or a list of layer diameters.")

        if len(D_layers) != Np.shape[0]:
            raise ValueError("The number of shell diameters must match the number of refractive-index layers.")

    if any(layer.size != 1 for layer in D_layers):
        raise ValueError(
            "This function expects a single particle with one diameter per shell. "
            "Use the ensemble functions for polydisperse size distributions."
        )

    D_shells = _np.asarray([float(layer[0]) for layer in D_layers], dtype=float)
    return wavelength, Nh, Np, D_layers, D_shells

@_hide_signature
def scatter_efficiency(wavelength: _Union[float, _np.ndarray],
                       Nh: _Union[float, _np.ndarray],
                       Np: _Union[float, _np.ndarray],
                       D: _Union[float, _np.ndarray],
                       *,
                       nmax: int = None,
                       return_coeffs: bool = False,
                       check_inputs: bool = True
                       ):

    '''
    Compute mie scattering parameters for multi-shell spherical particle.

    Parameters
    ----------
    wavelength : ndarray or float
        wavelength (microns)

    Nh : ndarray or float
        Complex refractive index of host. If ndarray, its size must be equal to
        len(wavelength)
        
    Np : float, 1darray or list
        Complex refractive index of each shell layer. The number of elements
        must be equal to len(D). Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (length must match that of wavelength)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
    D : float or list
        Outter diameter of each shell's layer (microns). Options are:
            float: solid sphere
            list:  multilayered sphere

    nmax: int, optional  
        Number of mie scattering coefficients. Default None
    
    Returns
    -------
    Qabs : ndarray
        Absorption efficiency

    Qsca : ndarray
        Scattering efficiency 
    
    gcos : ndarray
        Asymmetry parameter
    '''
    qabs, qsca, gcos, state = _scatter_efficiency_state(
        wavelength, Nh, Np, D, nmax=nmax, check_inputs=check_inputs
    )
    if return_coeffs:
        return qabs, qsca, gcos, _np.asarray(state["an"]), _np.asarray(state["bn"])

    return qabs, qsca, gcos

def _scatter_efficiency_state(wavelength, Nh, Np, D, *, nmax=None, check_inputs=False):
    """Shared implementation behind ``scatter_efficiency``.

    Returns the usual ``(qabs, qsca, gcos)`` plus the full coefficient state,
    so callers that go on to evaluate a phase function can reuse the Mie
    coefficients instead of recomputing the series.
    """
    wavelength, Nh, Np, _, D_shells = _normalize_single_particle_inputs(
        wavelength, Nh, Np, D, check_inputs=check_inputs
    )
    m = (Np / Nh).transpose()
    R = D_shells / 2.0
    x = _host_size_parameter(wavelength, Nh, R)

    # Vectorized path: avoids per-wavelength Python loops.
    qext, qsca, gcos, _, _, _, an, bn, state = _cross_section_at_lam(
        m, x, nmax, return_state=True
    )
    qabs = qext - qsca
    qabs = _clip_small_negative(qabs, qext, qsca)
    state["qabs"] = qabs
    state["qsca"] = qsca
    state["gcos"] = gcos
    state["an"] = an
    state["bn"] = bn
    return qabs, qsca, gcos, state

@_hide_signature
def scatter_coefficients(wavelength: _Union[float, _np.ndarray],
                         Nh: _Union[float, _np.ndarray],
                         Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
                         D: _Union[float, _np.ndarray],
                         *,
                         nmax: int = None,
                         check_inputs: bool = True):

    '''
    Compute mie scattering coefficients an and bn for multi-shell spherical 
    object. Layers must be sorted from inner to outter diameter

    Parameters
    ----------
    wavelength : ndarray or float
        wavelengtgh (microns)
        
    Nh : ndarray or float
        Complex refractive index of host. If ndarray, its size must be equal to
        len(wavelength)
        
    Np : float, 1darray or list
        Complex refractive index of each shell layer. The number of elements
        must be equal to len(D). Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (length must match that of wavelength)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
    D : float or list
        Outter diameter of each shell's layer (microns). Options are:
            float: solid sphere
            list:  multilayered sphere

    nmax: int, optional  
        Number of mie scattering coefficients. Default None

    Returns
    -------
    an : ndarray
        Scatttering coefficient M function
    bn : ndarray
        Scattering coefficient N function
    '''
    wavelength, Nh, Np, _, D_shells = _normalize_single_particle_inputs(
        wavelength, Nh, Np, D, check_inputs=check_inputs
    )

    m = (Np / Nh).transpose()
    R = D_shells / 2.0
    x = _host_size_parameter(wavelength, Nh, R)

    # determine nmax
    if nmax is None :
        # define nmax according to B.R Johnson (1996)
        nmax = _default_nmax(x[:, -1])

    # Coefficients are computed in one batched call over all wavelengths.
    state = _get_coated_state(m, x, nmax, reconstruct_raw=True, raise_on_raw_overflow=True)
    an, bn = state["an"], state["bn"]
    
    return an.reshape(-1, nmax), bn.reshape(-1, nmax)

def _pi_tau_1n(theta, nmax):
    """
    Compute the scalar tesseral function π_1n(θ) and τ_1n(θ)
    The arrays start with n = 1

    Adapted from the miepython library: https://github.com/scottprahl/miepython
    Original Author: Scott Prahl
    Modifications by: Francisco Ramírez (2025)
    
    Parameters:
        theta (ndarray): Polar angle θ in radians.
        nmax (int): Max degree of the associated Legendre polynomial.
        
    Returns:
        ndarray: π_1n(θ) = P_n^1(cos𝜃) / sin𝜃.
        ndarray: τ_1n(θ) = d/d𝜃 P_n^1(cos𝜃).
    """
    mu = _np.cos(theta)  # x = cos(θ)

    pi_n = _np.zeros((nmax, len(mu)))
    tau = _np.zeros((nmax, len(mu)))

    pi_nm2 = 0
    pi_n[0] = _np.ones_like(mu)

    for n in range(1, nmax):
        tau[n - 1] =            n * mu * pi_n[n - 1] - (n + 1) * pi_nm2
        temp = pi_n[n - 1]
        pi_n [n    ] = ((2 * n + 1) * mu * temp        - (n + 1) * pi_nm2) / n
        pi_nm2 = temp

    return pi_n, tau

def _amplitudes_from_coeffs(theta, an, bn):
    """
    Synthesize scattering amplitudes S1(theta), S2(theta) from Mie
    coefficients an, bn via vectorized angular synthesis.

    Parameters
    ----------
    theta : 1D ndarray
        Scattering angle(s) [rad].
    an, bn : ndarray, shape (n_wavelengths, nmax)
        Mie coefficients (or their normalized alpha/beta counterparts).

    Returns
    -------
    S1, S2 : ndarray, shape (n_theta, n_wavelengths)
    """
    nmax = an.shape[1]
    pi_n, tau = _pi_tau_1n(theta, nmax)

    n = _np.arange(1, nmax + 1)
    scale = (2 * n + 1) / ((n + 1) * n)

    # - pi_n/tau have shape (nmax, n_theta)
    # - an/bn have shape (n_lambda, nmax)
    # The matrix products below evaluate all theta and wavelength combinations
    # in one shot, replacing a Python loop over angles.
    weighted_pi = scale[:, None] * pi_n
    weighted_tau = scale[:, None] * tau
    S1 = weighted_pi.T @ an.T + weighted_tau.T @ bn.T
    S2 = weighted_tau.T @ an.T + weighted_pi.T @ bn.T
    return S1, S2

@_hide_signature
def scatter_amplitude(wavelength: _Union[float, _np.ndarray],
                      Nh: _Union[float, _np.ndarray], 
                      Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],  
                      D: _Union[float, _List[float]],
                      *,
                      theta: _Union[float, _np.ndarray] = None, 
                      nmax: int = None,
                      an: _np.ndarray = None,
                      bn: _np.ndarray = None,
                      check_inputs: bool = True):
    """
    Calculate the elements S1 (S11) and S2 (S22) of the scattering matrix for spheres.
    * For spheres S12 = S21 = 0

    The amplitude functions have been normalized so that when integrated
    over all 4*_pi solid angles, the integral will be qext*_pi*x**2.

    Adapted from the miepython library: https://github.com/scottprahl/miepython
    Original Author: Scott Prahl
    Modifications by: Francisco Ramírez (2025)

    Parameters:
        wavelength (ndarray or float): wavelengtgh (microns)
        
        Nh (ndarray or float): Complex refractive index of host. If 
                                   ndarray, len = wavelength
        
        Np (float, 1darray or list): Complex refractive index of each 
                                            shell layer. The number of elements 
                                            must be equal to len(D). 
            Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (len = wavelength)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
        D (float or list): Outter diameter of each shell's layer (microns). 
            Options are:
            float: solid sphere
            list:  multilayered sphere
        
        theta (ndarray or float): Scattering angle (radians). Default None
        
        nmax (int, optional): Number of mie scattering coefficients. Default None

        check_inputs (bool): True if user wants to check the inputs. Default True

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    # first check inputs and arrange them in np arrays
    if check_inputs:
        wavelength, Nh, Np, D, _ = _check_mie_inputs(wavelength, Nh, Np, D)
    
    # checks variable theta
    theta = _check_theta(theta)

    # Accept optional precomputed coefficients so callers can reuse them and skip
    # redundant coefficient recomputation.
    if an is None or bn is None:
        an, bn = scatter_coefficients(wavelength, Nh, Np, D,
                                      nmax=nmax,
                                      check_inputs=False)
    else:
        an = _np.asarray(an)
        bn = _np.asarray(bn)
        if an.ndim != 2 or bn.ndim != 2:
            raise ValueError("an and bn must have shape (n_wavelengths, nmax).")
        if an.shape != bn.shape:
            raise ValueError("an and bn must have the same shape.")
        if an.shape[0] != len(wavelength):
            raise ValueError("an and bn first dimension must match len(wavelength).")

    S1, S2 = _amplitudes_from_coeffs(theta, an, bn)

    return S1, S2

@_hide_signature
def scatter_stokes(wavelength: _Union[float, _np.ndarray], 
                   Nh: _Union[float, _np.ndarray], 
                   Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
                   D: _Union[float, _List[float]],
                   *,
                   theta: _Union[float, _np.ndarray] = None, 
                   nmax: int = None,
                   check_inputs: bool = True):
    """
    Calculate the Stokes parameters S11, S12, S33 and S34 of a sphere. 

    Parameters:
        theta (ndarray or float): Scattering angle (radians)

        wavelength (ndarray or float): wavelengtgh (microns)
        
        Nh (ndarray or float): Complex refractive index of host. If 
                                   ndarray, len(Nh) == len(wavelength)

        Np (float, 1darray or list): Complex refractive index of each 
                                            shell layer. Np.shape[1] == len(D). 
            Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (len = wavelength)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
        D (float or list): Outter diameter of each shell's layer (microns). 
            Options are:
            float: solid sphere
            list:  multilayered sphere

        nmax (int, optional): Number of mie scattering coefficients. Default None

        as_ndarray (bool): True if user wants the output as ndarray. Otherwise, 
        the output is a pd.DataFrame. Default False

    Returns:
        phase_fun: the scattering phase function (as pd.DataFrame or ndarray).
        If DataFrame, index is theta in radians.
    """

    # Organize D format
    if check_inputs:
        wavelength, Nh, Np, D, _ = _check_mie_inputs(wavelength, Nh, Np, D)
    
    # checks variable theta
    theta = _check_theta(theta)
    
    # Get scattering amplitude elements S1 and S2
    s1, s2 = scatter_amplitude(wavelength, Nh, Np, D,
                               theta=theta,
                               nmax=nmax,
                               check_inputs=False)

    # Compute stokes parameters
    S11 =1/2*(_np.abs(s1)**2 + _np.abs(s2)**2)
    S12 =1/2*(_np.abs(s1)**2 - _np.abs(s2)**2)
    S33 =1/2*(s2.conj()*s1 + s2*s1.conj())
    S34 =1j/2*(s2.conj()*s1 - s2*s1.conj())

    return S11, S12, S33, S34

def _phase_function_single(wavelength: _Union[float, _np.ndarray], 
                            Nh: _Union[float, _np.ndarray], 
                            Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
                            D: _Union[float, _List[float]],
                            *,
                            theta: _Union[float, _np.ndarray] = None, 
                            nmax: int = None, 
                            an: _np.ndarray = None,
                            bn: _np.ndarray = None,
                            mu_weights: _Optional[_np.ndarray] = None,
                            as_ndarray: bool = False,
                            check_inputs: bool = True):
    """
    Calculate the scattering phase function of a single sphere. The intensity
    is normalized such that the integral is equal to qsca.

    Adapted from the miepython library: https://github.com/scottprahl/miepython
    Original Author: Scott Prahl
    Modifications by: Francisco Ramírez (2025)

    Parameters:
        theta (ndarray or float): Scattering angle (radians)

        wavelength (ndarray or float): wavelengtgh (microns)
        
        Nh (ndarray or float): Complex refractive index of host. If 
                                   ndarray, len(Nh) == len(wavelength)

        Np (float, 1darray or list): Complex refractive index of each 
                                            shell layer. Np.shape[1] == len(D). 
            Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (len = wavelength)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
        D (float or list): Outter diameter of each shell's layer (microns). 
            Options are:
            float: solid sphere
            list:  multilayered sphere

        nmax (int, optional): Number of mie scattering coefficients. Default None

        an: either a coefficient-state dict (as produced by
        ``_scatter_efficiency_state``) to reuse, or a raw ``an`` array. If
        omitted the state is computed here.

        mu_weights (ndarray, optional): quadrature weights matching ``theta``
        on mu=cos(theta), ascending. Supply these whenever ``theta`` is not a
        (near-)uniform grid -- Gauss-Legendre abscissas in particular -- so the
        qsca renormalization below uses the quadrature that is correct for the
        grid instead of Simpson's rule, which assumes uniform spacing and is
        off by ~10% on Legendre abscissas.

        as_ndarray (bool): True if user wants the output as ndarray. Otherwise,
        the output is a pd.DataFrame. Default False

    Returns:
        phase_fun: the scattering phase function (as pd.DataFrame or ndarray)
    """
    # Organize D format
    wavelength, Nh, Np, _, D_shells = _normalize_single_particle_inputs(
        wavelength, Nh, Np, D, check_inputs=check_inputs
    )

    # checks variable theta
    theta = _check_theta(theta)

    # Single code path: always work from the coefficient state.  Computing it
    # here when no cache was supplied both avoids the overflow-prone raw
    # an/bn reconstruction that scatter_coefficients() insists on, and keeps
    # the qsca normalization identical however this function is reached.
    coeff_state = an if isinstance(an, dict) else None
    if coeff_state is None:
        _, _, _, coeff_state = _scatter_efficiency_state(
            wavelength, Nh, Np, D_shells, nmax=nmax, check_inputs=False
        )

    coeff_a = coeff_state.get("an")
    coeff_b = coeff_state.get("bn")
    if (
        coeff_a is None
        or coeff_b is None
        or not _np.all(_np.isfinite(coeff_a))
        or not _np.all(_np.isfinite(coeff_b))
    ):
        # Raw coefficients overflowed (large complex host size parameter);
        # the normalized alpha/beta carry the same angular shape safely.
        coeff_a = coeff_state["alpha"]
        coeff_b = coeff_state["beta"]
    coeff_a = _np.asarray(coeff_a, dtype=_np.complex128)
    coeff_b = _np.asarray(coeff_b, dtype=_np.complex128)
    if coeff_a.ndim == 1:
        coeff_a = coeff_a.reshape(1, -1)
        coeff_b = coeff_b.reshape(1, -1)
    s1, s2 = _amplitudes_from_coeffs(theta, coeff_a, coeff_b)
    phase_fun = (_np.abs(s1) ** 2 + _np.abs(s2) ** 2) / 2

    # Renormalize the angular shape so 2*pi*int(p dmu) == qsca.  This is what
    # makes the phase function consistent with the Johnson absorbing-host
    # cross sections, where the plain 1/(pi*(k*D/2)^2) scale factor is wrong.
    q_target = _np.asarray(coeff_state.get("qsca", _np.ones(wavelength.size)), dtype=float)
    mu = _np.cos(theta)
    order = _np.argsort(mu)
    if mu_weights is None:
        q_shape = 2.0 * _pi * _simpson(phase_fun[order, :], mu[order], axis=0)
    else:
        w = _np.asarray(mu_weights, dtype=float)
        if w.shape != mu.shape:
            raise ValueError("mu_weights must have the same length as theta.")
        q_shape = 2.0 * _pi * (w[order] @ phase_fun[order, :])
    with _np.errstate(divide='ignore', invalid='ignore'):
        norm = _np.divide(
            q_target,
            q_shape,
            out=_np.zeros_like(q_target, dtype=float),
            where=(q_target > 0.0) & _np.isfinite(q_shape) & (q_shape > 0.0),
        )
    phase_fun = phase_fun * norm.reshape(1, -1)

    # return phase function as ndarray
    if as_ndarray: return phase_fun

    # if not convert phase function to dataframe
    df_phase_fun = _pd.DataFrame(data=phase_fun, 
                                 index=_pd.Index(_np.degrees(theta), 
                                                 name='Theta (deg)'), 
                                 columns=wavelength)

    return df_phase_fun

@_hide_signature
def phase_scatt_HG(wavelength: _Union[float, _np.ndarray], 
                   gcos: _Union[float, _np.ndarray], 
                   qsca: _Union[float, _np.ndarray] = 1,
                   *,
                   theta: _Union[float, _np.ndarray] = None, 
                   as_ndarray: bool = False):
    """
    Compute the Heyney-Greenstein phase function

    Parameters
        wavelength : ndarray or float
            wavelengtgh (microns)

        gcos : float or ndarray
            Asymmetry parameter
        
        qsca: float or ndarray (optional)
            Scattering efficiency. If 1, then integral of phase function = 1.
            Default 1
        
        theta : ndarray or float (optional)
            Scattering angle (radians). If None, then 0 to 2*_pi in 1 degree steps.
            Default None
        
        as_ndarray : bool (optional)
            True if user wants the output as ndarray. Otherwise, the output is a pd.DataFrame. Default False

    Return
        p_theta_HG: float or ndarray
            Phase function
    """
    wavelength = _np.atleast_1d(_np.asarray(wavelength, dtype=float))
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError("wavelength must be a non-empty 1D array.")

    gcos = _as_1d_array(gcos, "gcos", n_wavelengths=wavelength.size, dtype=float)
    qsca = _as_1d_array(qsca, "qsca", n_wavelengths=wavelength.size, dtype=float)
    theta = _check_theta(theta)

    gg, tt = _np.meshgrid(gcos, theta)

    p_theta_HG = 1/(4*_np.pi)*(1 - gg**2)/(1 + gg**2 - 2*gg*_np.cos(tt))**(3/2)

    p_theta_HG = p_theta_HG * qsca.reshape(1, -1)

    # return phase function as ndarray
    if as_ndarray: return p_theta_HG

    # if not convert phase function to dataframe
    df_phase_fun = _pd.DataFrame(data=p_theta_HG, 
                                index=_pd.Index(theta, 
                                                name='Theta (rad)'), 
                                columns=wavelength,)

    return df_phase_fun
    
@_hide_signature
def scatter_from_phase_function(phase_fun,
                                *,
                                atol_deg: float = 1.0,
                                ) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Compute Qsca and <cos theta> from a DataFrame whose rows are labeled
    with scattering angles and columns with wavelengths.

    Integration is performed with Simpson's rule over mu = cos(theta), after
    sorting samples into ascending mu order.

    Parameters
    ----------
    phase_fun : pd.DataFrame
        Phase function. Row index must be theta in radians or degrees
        (not necessarily uniform). Columns correspond to different wavelengths.

    atol_deg : float
        Tolerance in degrees for verifying that the input theta grid covers
        the needed range [0, pi] (radians) or [0°, 180°] (degrees).
        Default is 1 degree.

    Returns
    -------
    qsca : ndarray
        Scattering efficiency for each column.
    
    gcos : ndarray
        Asymmetry parameter for each column.
    """
    if not isinstance(phase_fun, _pd.DataFrame):
        raise TypeError("phase_fun must be a pandas DataFrame (angles as index in radians or degrees).")

    # ---- 1) Sort and basic checks ----
    pf = phase_fun.sort_index().copy()
    # ensure numeric index (angles)
    try:
        theta_all = _np.asarray(pf.index, dtype=float)
    except Exception as e:
        raise TypeError("Row index must be numeric angles (float) in radians or degrees.") from e

    if theta_all.size < 2:
        raise ValueError("Theta index must contain at least two samples.")

    # Accept either radians [0, pi] or degrees [0, 180]. Use range-based
    # detection to preserve backward compatibility with degree-indexed inputs.
    atol_rad = _np.radians(atol_deg)
    theta_max = float(_np.max(theta_all))
    theta_min = float(_np.min(theta_all))
    is_radians = bool(theta_max <= (2*_np.pi + atol_rad) and theta_min >= -atol_rad)

    if is_radians:
        clip_lo, clip_hi = -atol_rad, _np.pi + atol_rad
    else:
        clip_lo, clip_hi = -atol_deg, 180.0 + atol_deg

    # ---- 2) Clip to the integration range [0, pi] or [0°, 180°] ----
    pf = pf.loc[(pf.index >= clip_lo) & (pf.index <= clip_hi)].copy()

    # Consolidate duplicates by averaging
    pf.index = _np.asarray(pf.index, dtype=float)
    pf = pf.groupby(level=0).mean().sort_index()

    # Re-check span after clipping/cleaning
    theta_clipped = pf.index.to_numpy()
    if theta_clipped.size < 2:
        raise ValueError("Not enough theta samples in the valid integration range after clipping/cleaning.")
    
    # Coverage near 0 and 180 is required for interpolation-based paths.
    # Direct Gauss matching does not need endpoint samples.
    if is_radians:
        has_endpoint_coverage = bool(theta_clipped[0] <= atol_rad and
                                     theta_clipped[-1] >= _np.pi - atol_rad)
    else:
        has_endpoint_coverage = bool(theta_clipped[0] <= atol_deg and
                                     theta_clipped[-1] >= 180.0 - atol_deg)

    if not has_endpoint_coverage:
        if is_radians:
            raise ValueError(
                f"Theta index must cover at least [0, pi] within ±{atol_rad:.5f} rad for this integration path. "
                f"Got range [{theta_clipped[0]:.5f}, {theta_clipped[-1]:.5f}] rad."
            )
        raise ValueError(
            f"Theta index must cover at least [0°, 180°] within ±{atol_deg}° for this integration path. "
            f"Got range [{theta_clipped[0]:.3f}°, {theta_clipped[-1]:.3f}°]."
        )

    theta_rad = theta_clipped if is_radians else _np.radians(theta_clipped)
    mu = _np.cos(theta_rad)
    order = _np.argsort(mu)
    mu_sorted = mu[order]
    p_eval = pf.to_numpy(dtype=float, copy=False)
    p_sorted = p_eval[order, :]

    qsca = 2.0 * _np.pi * _simpson(p_sorted, mu_sorted, axis=0)
    with _np.errstate(divide='ignore', invalid='ignore'):
        gcos = (2.0 * _np.pi * _simpson((mu_sorted[:, None] * p_sorted), mu_sorted, axis=0)) / qsca
    
    # ---- 6) Sanitize bad/zero cases ----
    mask_bad = ~_np.isfinite(qsca) | (qsca <= 0.0)
    if _np.any(mask_bad):
        qsca[mask_bad] = 0.0
        gcos[mask_bad] = 0.0

    return qsca, gcos

@_hide_signature
def phase_scatt_ensemble(wavelength: _Union[float, _np.ndarray],
                        theta: _Union[float, _np.ndarray],
                        Nh: _Union[float, _np.ndarray],
                        Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
                        D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
                        fv: float = 0.0,
                        *, 
                        size_dist: _np.ndarray = None,
                        nmax: int = None,
                        coeff_cache: _Optional[_List[_Tuple[_np.ndarray, _np.ndarray]]] = None,
                        mu_weights: _Optional[_np.ndarray] = None,
                        as_ndarray: bool = False,
                        check_inputs: bool = True,
                        effective_medium: bool = False,
                        dependent_scatt: bool = False):
    """
    Calculate the scattering phase function for multiple hard-spheres under unpolarized light. 
    The intensity is normalized such that the integral is equal to qsca

    Parameters:
    ----------
    wavelength : ndarray or float 
        Wavelengtgh (microns)
    
    Nh : ndarray or float 
        Complex refractive index of host. If ndarray, len(Nh) == len(wavelength)

    Np (float, 1darray or list): Complex refractive index of each 
                                        shell layer. Np.shape[1] == len(D). 
        Options are:
        float:   solid sphere and constant refractive index
        1darray: solid sphere and spectral refractive index (len = wavelength)
        list:    multilayered sphere (with both constant or spectral refractive indexes)
    
    D : float, _np.ndarray or list
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        if multilayer sphere, use list of floats (monodisperse) or arrays (polydisperse).
    
    fv: float
        Filling fraction. Defaul 0.0
    
    size_dist: ndarray
        Diameter density distribution. len(size_dist) == len(D)
    
    theta : float or ndarray (optional)
        Scatttering angle (radians). Default None
    
    nmax : int (optional)
        Number of mie scattering coefficients. Default None

    coeff_cache : list[(an, bn)], optional
        Optional precomputed Mie coefficients per size bin. When provided,
        phase-function evaluation reuses coefficients and avoids recomputing
        them inside scatter_amplitude for each bin.

    mu_weights : ndarray, optional
        Quadrature weights on mu=cos(theta) matching `theta`. Required for
        correctness whenever `theta` is not (near-)uniformly spaced: each size
        bin is renormalized so 2*pi*int(p dmu) equals that bin's qsca, and on
        Gauss-Legendre abscissas the default Simpson rule misestimates that
        integral by ~10%. For a polydisperse ensemble the error does not
        cancel -- it reweights the bins relative to one another, making the
        result depend on the number of quadrature nodes. Default None
        (Simpson, appropriate for uniform theta grids).

    as_ndarray : bool (optional)
        True if user wants the output as ndarray. Otherwise, the output is a pd.DataFrame. 
        Default False
    
    check_inputs : bool (optional)
        If True, check mie scattering inputs. Default True
        
    effective_medium : bool (optional)
        If True, compute the effective refractive index of the host using Bruggeman EMT.
        Default False
    
    dependent_scatt : bool (optional)
        If True, include structure factor in phase function calculation. Default False

    Returns:
    ----------
    phase_fun: the scattering phase function (as pd.DataFrame or ndarray)
    """
    # Input checks
    # Preserve whatever the caller passed for size_dist (ndarray, None, or a
    # closed-form distribution object from empylib.dense_spheres) so that,
    # even after check_inputs resolves it to a concrete array below for this
    # function's own per-bin loop, the ORIGINAL value is still available to
    # forward to structure_factor_PY's analytic fast path in the
    # dependent_scatt branch further down.
    orig_size_dist = size_dist
    if check_inputs:
            wavelength, Nh, Np, D, size_dist = _check_mie_inputs(wavelength, Nh, Np, D,
                                                         size_dist=size_dist)
    elif size_dist is not None and not isinstance(size_dist, _np.ndarray):
        # check_inputs=False internal-forwarding path (e.g. called from
        # cross_section_ensemble): D is already concrete at this point, so
        # just resolve local weights for this function's own per-bin loop.
        _w = size_dist.pdf(_np.asarray(D[-1], dtype=float))
        size_dist = _w / _w.sum()

    # asses if fv is within 0 and 1
    if not (0 <= fv < 1):
        raise ValueError("Filling fraction fv must be in the range [0, 1).")

    # checks variable theta
    theta = _check_theta(theta)

    if effective_medium and fv > 0:
        Nh = _effective_host(fv, Np, Nh, D, size_dist,
                            emt_multilayer_fn=_emt_multilayer_sphere,
                            emt_brugg_fn=_emt_brugg)

    # Get form factor
    if size_dist is None:
        # Monodisperse
        # Reuse cached coefficients if available (index 0 for single bin).
        an_bin = bn_bin = None
        if coeff_cache is not None and len(coeff_cache) > 0:
            if isinstance(coeff_cache[0], dict):
                an_bin = coeff_cache[0]
                bn_bin = None
            else:
                an_bin, bn_bin = coeff_cache[0]
        phase_fun = _phase_function_single(wavelength, Nh, Np, D,
                                         theta=theta,
                                         nmax=nmax,
                                         an=an_bin,
                                         bn=bn_bin,
                                         mu_weights=mu_weights,
                                         as_ndarray=True,
                                         check_inputs=False)
    else:
        Ac = _np.pi*(D[-1]/2)**2  # cross-sectional area of each sphere

        # Polydisperse: ensemble average over diameter distribution
        phase_fun = _np.zeros((len(theta), len(wavelength)), dtype=float)
        for i in range(len(size_dist)):
            Di = [d[i] for d in D]  # diameter of each layer for current size bin
            # Reuse per-bin coefficients computed upstream in cross_section_ensemble.
            an_bin = bn_bin = None
            if coeff_cache is not None and i < len(coeff_cache):
                if isinstance(coeff_cache[i], dict):
                    an_bin = coeff_cache[i]
                    bn_bin = None
                else:
                    an_bin, bn_bin = coeff_cache[i]
            # For each diameter, compute phase function
            phase_fun += size_dist[i] * Ac[i] * _phase_function_single(wavelength, Nh, Np, Di,
                                                                     theta=theta,
                                                                     nmax=nmax,
                                                                     an=an_bin,
                                                                     bn=bn_bin,
                                                                     mu_weights=mu_weights,
                                                                     as_ndarray=True,
                                                                     check_inputs=False)
        
        # Normalize by average cross-sectional area
        phase_fun /= _np.sum(size_dist * Ac)

    if dependent_scatt:
        # Get structure factor. Forward orig_size_dist (not the concrete
        # array resolved above) so a closed-form distribution reaches
        # structure_factor_PY's analytic fast path.
        S_q = structure_factor_PY(wavelength, Nh, D, fv,
                                theta=theta,
                                size_dist=orig_size_dist,
                                check_inputs=False)

        phase_fun = phase_fun*S_q

    # return phase function as ndarray
    if as_ndarray:
        return phase_fun

    # if not convert phase function to dataframe
    df_phase_fun = _pd.DataFrame(data=phase_fun,
                                 index=_pd.Index(theta, name='Theta (rad)'),
                                 columns=wavelength)

    return df_phase_fun

def _gauss_legendre_mu(nquad):
    """Gauss-Legendre nodes/weights on mu in [-1, 1], sorted ascending in mu."""
    mu, w = _leggauss(int(nquad))
    order = _np.argsort(mu)
    return mu[order], w[order]

# Gauss-Legendre nodes per unit size parameter for the ensemble angular pass.
#
# A Mie phase function oscillates ~x times across theta in [0, pi], so the
# quadrature must resolve O(x) features and the node count has to grow
# LINEARLY with the size parameter.  The older sqrt(x) heuristic
# (_estimate_theta_npts) silently loses accuracy as particles grow: measured
# csca error 3.5e-5 at x~47, 2.4e-2 at x~149 and 3.2e-1 at x~361.  At 3 nodes
# per unit x the error stays ~2e-5 across that whole range, for roughly twice
# the angular work of the old heuristic at large x -- which the removal of the
# second angular pass more than pays for.
_GL_NODES_PER_SIZE_PARAM = 3.0
# Guard against a pathological allocation if a caller passes an absurd
# wavelength/diameter combination.
_GL_NODES_MAX = 20000

def _quadrature_order(wavelength, Nh, D, size_dist=None, n_moments=None, n_quad=None):
    """Number of Gauss-Legendre nodes for the ensemble angular integration.

    Floored by 4*n_moments so the quadrature can resolve every requested
    Legendre polynomial exactly, by the caller's `n_quad` request, and by
    `_GL_NODES_PER_SIZE_PARAM * x_max` so forward-peaked phase functions at
    large size parameters stay resolved.
    """
    wavelength_arr = _np.atleast_1d(_np.asarray(wavelength, dtype=float))
    Nh_arr = _np.atleast_1d(_np.asarray(Nh, dtype=complex))
    # D=None is valid when size_dist is a closed-form distribution object (the
    # per-bin diameters are generated downstream via size_dist.discretize()).
    # Fall back to the distribution's mean diameter so this heuristic does not
    # see a bare `None` and produce NaN.
    D_for_heuristic = D
    if D_for_heuristic is None and size_dist is not None and hasattr(size_dist, "D_mean"):
        D_for_heuristic = size_dist.D_mean
    D_arr = _np.atleast_1d(_np.asarray(
        D_for_heuristic[-1] if isinstance(D_for_heuristic, list) else D_for_heuristic,
        dtype=float))

    x_max = float(_np.max(_np.pi * _np.real(Nh_arr) * _np.max(D_arr) / wavelength_arr))
    floor = int(_np.ceil(_GL_NODES_PER_SIZE_PARAM * max(x_max, 0.0)))
    floor = max(floor, 8)
    if n_moments is not None:
        floor = max(floor, 4*int(n_moments))
    if n_quad is not None:
        floor = max(floor, int(n_quad))
    return int(min(floor, _GL_NODES_MAX))

def _ensemble_optics(wavelength, Nh, Np, D, fv=0.0, *,
                     size_dist=None,
                     dependent_scatt=False,
                     effective_medium=False,
                     n_moments=None,
                     n_quad=None,
                     nmax=None,
                     check_inputs=True):
    """Single-pass optical properties of a sphere ensemble.

    This is the one place that owns the angular grid. It runs the per-size-bin
    Mie coefficient loop once, evaluates the ensemble phase function once on a
    single Gauss-Legendre mu grid, and derives everything else from weighted
    sums against that one grid:

        csca  <- a_0        (2*pi*int p dmu, times the mean geometric area)
        gcos  <- a_1 / a_0  (<cos theta>)
        a_l   <- Legendre moments for iadpython's pf_type='MOMENTS'

    Deriving `csca`/`gcos` from the same quadrature that produces the moments
    replaces a separate uniform-theta Simpson pass, and is markedly more
    accurate: against a dense reference, Gauss-Legendre reaches ~3e-5 relative
    error on qsca where 101-point Simpson reaches ~2e-3.

    The angular pass runs only when it is actually needed -- that is, when the
    structure factor modifies the angular distribution (`dependent_scatt`) or
    Legendre moments were requested (`n_moments`). Otherwise the direct,
    numerically stable Mie sums remain the source of truth for csca and gcos.

    Returns
    -------
    dict with keys:
        cabs, csca, gcos : (nlam,) ndarray
        moments          : (n_moments, nlam) ndarray or None
        phase            : (n_quad, nlam) ndarray or None -- ensemble phase
                           function on the Gauss-Legendre grid
        mu, weights      : (n_quad,) ndarray or None
        coeff_cache      : list of per-bin Mie coefficient state
        Nh_eff           : (nlam,) ndarray -- host index actually used
        wavelength, Np, D, size_dist : the resolved/validated inputs, in the
                           shapes every `check_inputs=False` path expects
    """
    # Preserve the caller's size_dist (ndarray, None, or a closed-form
    # distribution object) so it can still reach structure_factor_PY's
    # analytic fast path even after validation resolves a concrete array.
    orig_size_dist = size_dist
    if check_inputs:
        wavelength, Nh, Np, D, size_dist = _check_mie_inputs(wavelength, Nh, Np, D,
                                                            size_dist=size_dist)
    elif size_dist is not None and not isinstance(size_dist, _np.ndarray):
        _w = size_dist.pdf(_np.asarray(D[-1], dtype=float))
        size_dist = _w / _w.sum()

    if not (0 <= fv < 1):
        raise ValueError("Filling fraction fv must be in the range [0, 1).")

    if effective_medium and fv > 0:
        Nh = _effective_host(fv, Np, Nh, D, size_dist,
                             emt_multilayer_fn=_emt_multilayer_sphere,
                             emt_brugg_fn=_emt_brugg)

    Ac = _np.pi*(D[-1]/2)**2                                   # geometric area per bin
    n_bins = 1 if size_dist is None else len(size_dist)
    p = _np.asarray([1.0]) if size_dist is None else size_dist

    # ---------- per-size-bin Mie coefficients (the expensive loop) ----------
    cabs_av = _np.zeros_like(wavelength, dtype=float)
    csca_av = _np.zeros_like(wavelength, dtype=float)
    gcos_av = _np.zeros_like(wavelength, dtype=float)
    coeff_cache = []
    for i in range(n_bins):
        Di = [d[i] for d in D]
        qabs, qsca, gcos, state = _scatter_efficiency_state(
            wavelength, Nh, Np, Di, nmax=nmax
        )
        coeff_cache.append(state)
        cabs_av += p[i] * qabs * Ac[i]
        csca_av += p[i] * qsca * Ac[i]
        gcos_av += p[i] * qsca * gcos * Ac[i]      # weighted by scattering

    with _np.errstate(divide='ignore', invalid='ignore'):
        gcos_av = _np.divide(gcos_av, csca_av,
                             out=_np.zeros_like(gcos_av),
                             where=~_np.isclose(csca_av, 0.0))

    result = {
        "cabs": cabs_av, "csca": csca_av, "gcos": gcos_av,
        "moments": None, "phase": None, "mu": None, "weights": None,
        "coeff_cache": coeff_cache, "Nh_eff": Nh,
        "wavelength": wavelength, "Np": Np, "D": D, "size_dist": size_dist,
    }

    need_angular = dependent_scatt or (n_moments is not None)
    if not need_angular:
        return result

    # ---------- the single angular pass ----------
    nquad = _quadrature_order(wavelength, Nh, D, size_dist=orig_size_dist,
                              n_moments=n_moments, n_quad=n_quad)
    mu, w = _gauss_legendre_mu(nquad)

    # Forward orig_size_dist so a closed-form distribution survives this
    # handoff and reaches structure_factor_PY's analytic fast path; D is
    # already concrete here.
    phase = phase_scatt_ensemble(wavelength, _np.arccos(mu), Nh, Np, D, fv,
                                 size_dist=orig_size_dist,
                                 nmax=nmax,
                                 coeff_cache=coeff_cache,
                                 mu_weights=w,
                                 as_ndarray=True,
                                 effective_medium=False,
                                 check_inputs=False,
                                 dependent_scatt=dependent_scatt)

    result["phase"] = phase
    result["mu"] = mu
    result["weights"] = w

    if dependent_scatt:
        # Re-integrate only when the structure factor has modified the angular
        # distribution. Without it the direct stable Mie sums above remain the
        # source of truth for Qsca and g.
        pw = phase * w[:, None]
        qsca_pf = 2.0 * _pi * pw.sum(axis=0)
        gcos_pf = 2.0 * _pi * (mu[:, None] * pw).sum(axis=0)
        with _np.errstate(divide='ignore', invalid='ignore'):
            gcos_av = _np.divide(gcos_pf, qsca_pf,
                                 out=_np.zeros_like(gcos_pf),
                                 where=~_np.isclose(qsca_pf, 0.0))
        bad = ~_np.isfinite(qsca_pf) | (qsca_pf <= 0.0)
        if _np.any(bad):
            qsca_pf = _np.where(bad, 0.0, qsca_pf)
            gcos_av = _np.where(bad, 0.0, gcos_av)
        # phase_scatt_ensemble returns the area-weighted mean phase function,
        # so 2*pi*int(p dmu) is the mean Qsca; scale by the mean area <A>.
        result["csca"] = qsca_pf * float(_np.sum(p * Ac))
        result["gcos"] = gcos_av

    if n_moments is not None:
        P_all = _np.array([_eval_legendre(l, mu) for l in range(int(n_moments))])
        a_raw = P_all @ (phase * w[:, None])            # (n_moments, nlam)
        a0 = a_raw[0, :]
        zero_mask = _np.isclose(a0, 0.0)
        out = _np.zeros_like(a_raw)
        if _np.any(~zero_mask):
            out[:, ~zero_mask] = a_raw[:, ~zero_mask] / a0[~zero_mask]
        result["moments"] = out

    return result

def phase_function_moments(wavelength: _Union[float, _np.ndarray],
                           Nh: _Union[float, _np.ndarray],
                           Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
                           D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
                           fv: float = 0.0,
                           *,
                           size_dist: _np.ndarray = None,
                           n_moments: int = None,
                           quad_pts: int = None,
                           dependent_scatt: bool = False,
                           nmax: int = None,
                           coeff_cache: _Optional[_List[_Tuple[_np.ndarray, _np.ndarray]]] = None,
                           effective_medium: bool = False,
                           check_inputs: bool = True):
    """
    Legendre moments a_l of the ensemble scattering phase function, via
    Gauss-Legendre quadrature evaluated directly on mu=cos(theta) -- no
    interpolation. Intended for feeding iadpython's `pf_type='MOMENTS'`
    Sample/Layer construction, bypassing the spline+quadrature its
    `pf_type='TABULATED'` path requires.

    Parameters:
    ----------
    n_moments : int, optional
        Number of Legendre moments a_0..a_{n_moments-1} to return.
        Provide at most one of `n_moments` or `quad_pts`. If neither is
        given, defaults to 33 (iadpython's own default of 2*quad_pts+1
        for quad_pts=16). For a mismatched-refractive-index multilayer
        stack in iadpython, size generously beyond 2*quad_pts+1 -- see
        iadpython.Layer's docstring note on pf_type='MOMENTS' -- which
        is exactly when an explicit `n_moments` (rather than `quad_pts`)
        is useful: it decouples "how many moments to compute" from
        "what quad_pts iadpython nominally uses" for that one Sample.

    quad_pts : int, optional
        Convenience alternative to `n_moments`: sets `n_moments =
        2*quad_pts + 1`, matching the count iadpython's `phase_legendre`
        needs for that many quadrature points. Mutually exclusive with
        `n_moments`.

    dependent_scatt : bool, optional
        Whether to include the Percus-Yevick structure factor S(q)
        (default False).

    coeff_cache : list, optional
        Per-size-bin Mie coefficient state to reuse (e.g. captured via
        cross_section_ensemble's out_coeff_cache), avoiding redundant
        Mie-coefficient computation when the cross sections were
        already computed separately for the same ensemble.

    Returns:
    ----------
    a_raw : np.ndarray, shape (n_moments, n_wavelength)
        Legendre moments, normalized so a_raw[0, :] == 1 (matching
        iadpython.legendre_coeffs_from_df's convention exactly). A
        wavelength column that is identically zero (matched-index /
        no-scattering limit) stays all-zero rather than raising a
        division-by-zero.
    """
    if n_moments is not None and quad_pts is not None:
        raise ValueError("Provide at most one of n_moments or quad_pts.")
    if n_moments is None:
        n_moments = 2*int(quad_pts) + 1 if quad_pts is not None else 33

    # Reusing a supplied coefficient cache means skipping _ensemble_optics'
    # own Mie loop; go straight to the angular pass in that case.
    if coeff_cache is not None:
        nquad = _quadrature_order(wavelength, Nh, D, size_dist=size_dist,
                                  n_moments=n_moments)
        mu, w = _gauss_legendre_mu(nquad)
        phase = phase_scatt_ensemble(wavelength, _np.arccos(mu), Nh, Np, D, fv,
                                     size_dist=size_dist, nmax=nmax,
                                     coeff_cache=coeff_cache, mu_weights=w,
                                     as_ndarray=True,
                                     check_inputs=check_inputs,
                                     effective_medium=effective_medium,
                                     dependent_scatt=dependent_scatt)
        P_all = _np.array([_eval_legendre(l, mu) for l in range(n_moments)])
        a_raw = P_all @ (phase * w[:, None])
        a0 = a_raw[0, :]
        zero_mask = _np.isclose(a0, 0.0)
        out = _np.zeros_like(a_raw)
        if _np.any(~zero_mask):
            out[:, ~zero_mask] = a_raw[:, ~zero_mask] / a0[~zero_mask]
        return out

    return _ensemble_optics(wavelength, Nh, Np, D, fv,
                            size_dist=size_dist,
                            dependent_scatt=dependent_scatt,
                            effective_medium=effective_medium,
                            n_moments=n_moments,
                            nmax=nmax,
                            check_inputs=check_inputs)["moments"]

@_hide_signature
def cross_section_ensemble(
    wavelength: _Union[float, _np.ndarray], 
    Nh: _Union[float, _np.ndarray], 
    Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
    D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
    fv: float = 0.0, 
    *,
    size_dist: _np.ndarray = None, 
    effective_medium: bool = False,
    dependent_scatt: bool = False,
    phase_function: bool = False,
    nmax: int = None,
    check_inputs: bool = True,
    n_theta: int = 101,
    out_coeff_cache: list = None,
):
    """
    Compute size-averaged scattering/absorption cross sections and asymmetry parameter
    for an ensemble of hard spheres under the independent-scattering assumption.
    Not valid for metallic spheres or high volume fractions where near-field coupling
    is important.

    Parameters
    ----------
    wavelength : array-like, shape (nλ,)
        Wavelengths [µm], strictly positive.

    Nh : float or array-like (nλ,)
        Host refractive index (can be complex). If array-like, length must equal len(wavelength).

    Np : float, 1darray or list 
        Complex refractive index of each shell layer. Np.shape[1] == len(D). 
        Options are:
        - float:   solid sphere and constant refractive index
        - 1darray: solid sphere and spectral refractive index (len = wavelength)
        - list:    multilayered sphere (with both constant or spectral refractive indexes)
    
    D : float, np.ndarray or list
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        if multilayer sphere, use list of floats (monodisperse) or list of arrays (polydisperse).
    
    fv : float, optional
        Particle volume fraction in (0, 1). Used only to compute an effective medium Nh via
        `nk.emt_brugg(fv, Np, Nh)`.
    
    size_dist : array-like (nD,), optional
        Diameter's size distribution (polydisperse case). Size must be equal to D. Sum must be 1
        within tolerance; will be renormalized if slightly off.

    effective_medium : bool, optional
        Whether to compute an effective host refractive index via Bruggeman EMT (default: False)

    dependent_scatt : bool, optional
        Whether to include dependent scattering effects via Percus-Yevick structure factor
        (default: False; not recommended for metallic spheres or high fv)    
    
    phase_function : bool, optional
        If True, also return the phase function DataFrame (default: False)
    
    nmax : int, optional
        Number of mie scattering coefficients (default: None, automatic).
        
    check_inputs : bool, optional
        Whether to check mie inputs (default: True)    

    n_theta : int, optional
        Number of scattering angles (default: 100).

    out_coeff_cache : list, optional
        If given a mutable empty list, it is filled in-place with the
        per-size-bin Mie coefficient state this function already computes
        internally (the same cache it uses for its own phase-function
        integration). Lets a caller reuse these coefficients in a *separate*
        subsequent call (e.g. `phase_function_moments`) without recomputing
        them. Default: None (no caching side effect).

    Returns
    -------
    cabs_av : _np.ndarray, shape (nλ,)
        Size-averaged absorption cross section per particle [µm²].

    csca_av : _np.ndarray, shape (nλ,)
        Size-averaged scattering cross section per particle [µm²].
    
    g_av : _np.ndarray, shape (nλ,)
        Size-averaged asymmetry parameter (⟨cosθ⟩).

    phase_fun_df : pd.DataFrame or None
        Scattering phase function (if `phase_function=True`), with index=θ° and columns=λ.
        Otherwise, None.
    """
    # The Mie loop, the angular grid and every quantity derived from it are
    # owned by _ensemble_optics, so csca/gcos come from the same single
    # Gauss-Legendre pass rather than a separate uniform-theta Simpson pass.
    optics = _ensemble_optics(wavelength, Nh, Np, D, fv,
                              size_dist=size_dist,
                              dependent_scatt=dependent_scatt,
                              effective_medium=effective_medium,
                              nmax=nmax,
                              check_inputs=check_inputs)

    if out_coeff_cache is not None:
        out_coeff_cache.extend(optics["coeff_cache"])

    cabs_av, csca_av, gcos_av = optics["cabs"], optics["csca"], optics["gcos"]

    if not phase_function:
        return cabs_av, csca_av, gcos_av, None

    # A tabulated phase function was explicitly requested. It is returned on a
    # uniform theta grid of `n_theta` points -- the documented index convention
    # callers (and scatter_from_phase_function, which needs theta=0/pi endpoint
    # coverage) rely on. This second angular pass therefore happens only on
    # explicit request, never on the internal path.
    theta = _np.linspace(0.0, _np.pi, n_theta)
    phase_fun_df = phase_scatt_ensemble(optics["wavelength"], theta,
                                        optics["Nh_eff"], optics["Np"],
                                        optics["D"], fv,
                                        size_dist=size_dist,
                                        nmax=nmax,
                                        coeff_cache=optics["coeff_cache"],
                                        as_ndarray=False,
                                        effective_medium=False,
                                        check_inputs=False,
                                        dependent_scatt=dependent_scatt)

    return cabs_av, csca_av, gcos_av, phase_fun_df

