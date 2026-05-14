# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:38:11 2021

@author: PanxoPanza
"""
import numpy as _np
from numpy import pi as _pi, exp as _exp, conj as _conj, imag as _imag, real as _real, sqrt as _sqrt
from scipy.special import jv as _jv, yv as _yv
from scipy.integrate import simpson as _simpson
from .nklib import emt_brugg as _emt_brugg, emt_multilayer_sphere as _emt_multilayer_sphere
from .utils import (
    _as_1d_array,
    _check_mie_inputs,
    _check_theta,
    _hide_signature,
    _effective_host,
)
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
    'cross_section_ensemble',
)


def _trapz(y, x, axis=-1):
    """Compat wrapper for NumPy trapezoidal integration across versions."""
    if hasattr(_np, "trapezoid"):
        return _np.trapezoid(y, x, axis=axis)
    return _np.trapz(y, x, axis=axis)

def _log_RicattiBessel(x,nmax,nmx):
    '''
    Computes the logarithmic derivatives of Ricatti-Bessel functions,
        Dn(x) = psi_n'(x) / psi_n(x),
        Gn(x) = chi_n'(x) / chi_n(x), and
        Rn(x) = psi_n(x)  / xi_n(x);
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
        Rn(x)
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

    # Get Rn(x) by upwards recurrence
    Rnx = _np.zeros((x.shape[0], x.shape[1], nmax), dtype=_np.complex128)
    for i in range(nmax):
        if i == 0:
            Rim1x = 0.5 * (1 - _exp(-2j * x))
        else:
            Rim1x = Rnx[:, :, i - 1]

        Rnx[:, :, i] = Rim1x * (Gnx[:, :, i] + (i + 1) / x) / (Dnx[:, :, i] + (i + 1) / x)

    # Exact fallback for x = pi*n on the real axis (avoids numerical cancellation).
    pi_mask = (_imag(x) == 0) & (_np.mod(_real(x), _pi) == 0)
    if _np.any(pi_mask):
        nu = (n + 1) + 0.5
        for iwl, ish in _np.argwhere(pi_mask):
            xval = _real(x[iwl, ish])
            py = _sqrt(0.5 * _pi * xval) * _jv(nu, xval)
            chy = _sqrt(0.5 * _pi * xval) * _yv(nu, xval)
            gsy = py + 1j * chy
            Rnx[iwl, ish, :] = py / gsy

    Dn = Dnx[:, :, n]
    Gn = Gnx[:, :, n]
    Rn = Rnx[:, :, n]
    if squeeze_out:
        return Dn[0], Gn[0], Rn[0]
    return Dn, Gn, Rn

def _recursive_ab(m, n, Dn, Gn, Rn, Dn1, Gn1, Rn1):
    """
    Compute multilayer Mie coefficients ``an`` and ``bn`` using Johnson-style
    layer recursion, vectorized over wavelength.

    Notes
    -----
    - Internal working shape is ``(n_wavelengths, n_layers, n_orders)``.
    - For single-wavelength/single-particle calls, inputs may arrive as 2D;
      they are temporarily promoted to 3D and squeezed back on return.
    """
    m = _np.asarray(m, dtype=_np.complex128)
    Dn = _np.asarray(Dn, dtype=_np.complex128)
    Gn = _np.asarray(Gn, dtype=_np.complex128)
    Rn = _np.asarray(Rn, dtype=_np.complex128)
    Dn1 = _np.asarray(Dn1, dtype=_np.complex128)
    Gn1 = _np.asarray(Gn1, dtype=_np.complex128)
    Rn1 = _np.asarray(Rn1, dtype=_np.complex128)

    # If we promoted scalar-style inputs to batched shape, restore on return.
    squeeze_out = False
    if Dn.ndim == 2:
        # Promote to 3D so one vectorized implementation covers both cases.
        Dn = Dn.reshape(1, Dn.shape[0], Dn.shape[1])
        Gn = Gn.reshape(1, Gn.shape[0], Gn.shape[1])
        Rn = Rn.reshape(1, Rn.shape[0], Rn.shape[1])
        Dn1 = Dn1.reshape(1, Dn1.shape[0], Dn1.shape[1])
        Gn1 = Gn1.reshape(1, Gn1.shape[0], Gn1.shape[1])
        Rn1 = Rn1.reshape(1, Rn1.shape[0], Rn1.shape[1])
        m = m.reshape(1, -1)
        squeeze_out = True

    n_layers = Dn.shape[1]
    # Start from the core boundary condition before shell-by-shell updates.
    an = _np.zeros((Dn.shape[0], n), dtype=_np.complex128)
    bn = _np.zeros((Dn.shape[0], n), dtype=_np.complex128)

    # Layer-by-layer Johnson recursion, vectorized over wavelength.
    for i in range(1, n_layers + 1):
        # Relative index jump at interface i-1 -> i.
        ratio = (m[:, i] / m[:, i - 1]).reshape(-1, 1)

        # Auxiliary interface terms in Johnson's recursion.
        Un = (Rn[:, i - 1, :] * Dn[:, i - 1, :] - an * Gn[:, i - 1, :]) / (Rn[:, i - 1, :] - an + 1E-10)
        Vn = (Rn[:, i - 1, :] * Dn[:, i - 1, :] - bn * Gn[:, i - 1, :]) / (Rn[:, i - 1, :] - bn + 1E-10)

        an = Rn1[:, i - 1, :] * (ratio * Un - Dn1[:, i - 1, :]) / (ratio * Un - Gn1[:, i - 1, :])
        bn = Rn1[:, i - 1, :] * (Vn - ratio * Dn1[:, i - 1, :]) / (Vn - ratio * Gn1[:, i - 1, :])

    if squeeze_out:
        return an[0], bn[0]
    return an, bn
        
def _get_coated_coefficients(m,x, nmax=None):
    '''
    Compute the mie coefficients an and bn using recursion algorithm from
    Johnson, Appl. Opt. 35, 3286 (1996).

    Parameters
    ----------
    m : 1D numpy array
        normalized refractive index of shell's layers
    x : 1D numpy array
        size parameter of shell's layers'
    nmax : int
        max number of expansion coefficients.

    Returns
    -------
    an : 1D numpy array (size nmax)
         mie coefficient for M function.
    bn : 1D numpy array (size nmax)
        mie coefficient for N function.
    phi : 1D numpy array (size nmax)
        1st order Bessel-Ricatti function (evaluated at ka).
    Dn1 : 1D numpy array (size nmax)
        Derivative of 1st order Bessel-Ricatti function (evaluated at ka).
    xi : 1D numpy array (size nmax)
        3rd order Bessel-Ricatti function (evaluated at ka).
    Gn1 : 1D numpy array (size nmax)
        Derivative of 2nd order Bessel-Ricatti function (evaluated at ka).

    '''
    x = _np.asarray(x)
    m = _np.asarray(m, dtype=_np.complex128)
    # If called for one spectrum only, squeeze outputs back to 1D vectors.
    squeeze_out = False

    if x.ndim == 1:
        x = x.reshape(1, -1)
        m = m.reshape(1, -1)
        squeeze_out = True
    elif x.ndim != 2:
        raise ValueError("x must be 1D or 2D.")

    if m.shape != x.shape:
        raise ValueError("m and x must have the same shape.")

    ka = x[:, -1] # size parameter of outer layer

    # define nmax according to B.R Johnson (1996)
    if nmax is None :
        # One global nmax for the whole wavelength batch keeps array shapes fixed.
        # Effective per-wavelength truncation is handled naturally by tiny high-order terms.
        nmax = int(_np.round(_np.max(_np.abs(ka)) + 4*_np.max(_np.abs(ka))**(1/3) + 2))
    
    #----------------------------------------------------------------------
    #       Computing an and bn (main part of this code)
    #----------------------------------------------------------------------
    
    mix = m*x               # Ni*k*ri
    mi1 = _np.concatenate((m, _np.ones((m.shape[0], 1), dtype=_np.complex128)), axis=1)
    mi1x = mi1[:, 1:]*x        # Ni+1*k*ri
    
    # Computation of Dn(z), Gn(z) and Rn(z)
    nmx = int(_np.round(max(nmax, _np.max(abs(m*x))) + 16))
    
    # Get Dn(mi*x), Gn(mi*x), Rn(mi*x) 
    Dn, Gn, Rn = _log_RicattiBessel(mix,nmax,nmx)
    
    # Get Dn(mi+1*x), Gn(mi+1*x), Rn(mi+1*x)
    Dn1, Gn1, Rn1 = _log_RicattiBessel(mi1x,nmax,nmx)
    
    # Get an and bn
    an, bn = _recursive_ab(mi1, nmax, Dn, Gn, Rn, Dn1, Gn1, Rn1)
    
    # ---------------------------------------------------------------------
    #       computing secondary paramters
    # ---------------------------------------------------------------------
    # Get Bessel-Ricatti functions and derivatives for last shell layer
    n = _np.array(range(1,nmax+1))
    nu = n + 0.5
    ka_2d = ka.reshape(-1, 1)
    phi = _np.sqrt(0.5*_pi*ka_2d)*_jv(nu.reshape(1, -1), ka_2d) # phi(n,ka)
    chi = _np.sqrt(0.5*_pi*ka_2d)*_yv(nu.reshape(1, -1), ka_2d) # chi(n,ka)
    xi  = phi + 1j*chi                    # xi(n,ka)

    Dy = Dn1[:, -1, :]
    Gy = Gn1[:, -1, :]
    if squeeze_out:
        return an[0], bn[0], phi[0], Dy[0], xi[0], Gy[0]
    return an, bn, phi, Dy, xi, Gy

def _cross_section_at_lam(m,x,nmax = None):
    '''
    NEED TO CHECK FLUCTUATION FOR LARGE PARTICLES (F. RAMIREZ 2024)
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
        y_max = _np.max(_np.abs(y))
        nmax = int(_np.round(y_max + 4*y_max**(1/3) + 2))

    #------------------------------------------------------------------
    # Single batched coefficient call for all wavelengths.
    #------------------------------------------------------------------
    (an, bn, py, Dy, xy, Gy) = _get_coated_coefficients(m,x,nmax)

    # Matched-index absorbing-host cases are especially sensitive to
    # cancellation in the Johnson formulas below.  If the full Mie series is
    # already numerically extinguished, snap the coefficients to zero so the
    # derived efficiencies follow that same limit consistently.
    coeff_floor = 1e-10
    coeff_mask = _np.maximum(_np.max(_np.abs(an), axis=1), _np.max(_np.abs(bn), axis=1)) < coeff_floor
    if _np.any(coeff_mask):
        an = an.copy()
        bn = bn.copy()
        an[coeff_mask, :] = 0.0
        bn[coeff_mask, :] = 0.0

    # Absorbing-host correction factor from Johnson (1996), evaluated only where needed.
    imy = 2*_imag(y)
    ft = _np.full_like(_real(y), 2.0, dtype=float)
    mask_abs_host = _imag(y) > 1E-8
    if _np.any(mask_abs_host):
        imy_sel = imy[mask_abs_host]
        ft[mask_abs_host] = _real(imy_sel**2/(1 + (imy_sel - 1)*_exp(imy_sel)))
    
    # arranging pre-computing constants
    n = _np.arange(1, nmax + 1, dtype=float).reshape(1, -1)
    
    #------------------------------------------------------------------
    # Extinction efficiency
    #------------------------------------------------------------------
    en = (2*n+1)*_imag((- 2j*py*_conj(py)*_imag(Dy)          \
                       + _conj(an)*_conj(xy)*py*Dy           \
                       - _conj(bn)*_conj(xy)*py*_conj(Gy)    \
                       + an*xy*_conj(py)*Gy                  \
                       - bn*xy*_conj(py)*_conj(Dy))          \
                       /y.reshape(-1, 1))
    q = _np.sum(en, axis=1)
    Qext = _real(1/_real(y)*ft*q)
    
    #------------------------------------------------------------------
    # Scattering efficiency
    #------------------------------------------------------------------
    en = (2*n+1)*_imag((+ _np.abs(an*xy)**2*Gy               \
                       - _np.abs(bn*xy)**2*_conj(Gy)         \
                       )/y.reshape(-1, 1))
    q = _np.sum(en, axis=1)
    Qsca = _real(1/_real(y)*ft*q)
    
    #------------------------------------------------------------------
    # Asymmetry parameter
    #------------------------------------------------------------------
    anp1 = _np.zeros_like(an, dtype=_np.complex128)
    bnp1 = _np.zeros_like(bn, dtype=_np.complex128)
    anp1[:, :nmax-1] = an[:, 1:] # a(n+1) coefficient
    bnp1[:, :nmax-1] = bn[:, 1:] # b(n+1) coefficient

    asy1 = n*(n + 2)/(n + 1)*(an*_conj(anp1)+ bn*_conj(bnp1)) \
         + (2*n + 1)/(n*(n + 1))*_real(an*_conj(bn))
    
    asy2 = (2*n+1)*(an*_conj(an) + bn*_conj(bn))
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
    f = (2*n+1)*((-1)**n)*(an - bn)
    q = _np.sum(f, axis=1)
    Qb = _real(q*_conj(q)/y**2)
    
    #------------------------------------------------------------------
    # Forward scattering (not valid for absorbing host media)
    #------------------------------------------------------------------
    f = (2*n+1)*(an + bn)
    q = _np.sum(f, axis=1)
    Qf = _real(q*_conj(q)/y**2)
    
    #------------------------------------------------------------------
    # Condition outputs to avoid unphysical results
    #------------------------------------------------------------------
    if _np.any(coeff_mask):
        Qext[coeff_mask] = 0.0
        Qsca[coeff_mask] = 0.0
        Asym[coeff_mask] = 0.0

    Qsca = _np.where(Qsca < 0, 0, Qsca)
    Qext = _np.where(Qext < Qsca, Qsca, Qext)
    Asym = _np.clip(Asym, -1, +1)

    if squeeze_out:
        return Qext[0], Qsca[0], Asym[0], Qb[0], Qf[0], nmax, an[0], bn[0]
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
        wavelength, Nh, Np, D, _ = _check_mie_inputs(wavelength, Nh, Np, D)

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
    wavelength, Nh, Np, _, D_shells = _normalize_single_particle_inputs(
        wavelength, Nh, Np, D, check_inputs=check_inputs
    )

    m = (Np / Nh).transpose()
    R = D_shells / 2.0
    # Keep the full complex contrast m so matched absorbing media still reduce
    # to zero scattering, but build the size parameter from the real host
    # transport wavenumber so the Johnson-series normalization stays stable.
    kh = 2 * _pi * _real(Nh) / wavelength
    x = _np.outer(kh, R)
    
    # Vectorized path: avoids per-wavelength Python loops.
    qext, qsca, gcos, _, _, _, an, bn = _cross_section_at_lam(m, x, nmax)

    # outputs: qabs, qsca, gcos
    qabs = _np.maximum(qext - qsca, 0.0)
    if return_coeffs:
        return qabs, qsca, gcos, _np.asarray(an), _np.asarray(bn)

    return qabs, qsca, gcos

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
    # See scatter_efficiency(): use the transport wavenumber in x while
    # preserving the full complex material contrast in m.
    kh = 2 * _pi * _real(Nh) / wavelength
    x = _np.outer(kh, R)

    # determine nmax
    if nmax is None :
        y = _np.max(_np.abs(x[:, -1])) # largest size parameter of outer layer
        # define nmax according to B.R Johnson (1996)
        nmax = int(_np.round(y + 4*y**(1/3) + 2))

    # Coefficients are computed in one batched call over all wavelengths.
    an, bn, *_ = _get_coated_coefficients(m, x, nmax)
    
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

    _pi  = _np.zeros((nmax, len(mu)))
    tau = _np.zeros((nmax, len(mu)))
    
    pi_nm2 = 0
    _pi[0] = _np.ones_like(mu)
    
    for n in range(1, nmax):
        tau[n - 1] =            n * mu * _pi[n - 1] - (n + 1) * pi_nm2
        temp = _pi[n - 1]
        _pi [n    ] = ((2 * n + 1) * mu * temp        - (n + 1) * pi_nm2) / n
        pi_nm2 = temp
        
    return _pi, tau

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
    nmax = an.shape[1]

    # get _pi and tau angular functions
    _pi, tau = _pi_tau_1n(theta, nmax)

    # set scale for summation
    n = _np.arange(1, nmax + 1)
    scale = (2 * n + 1) / ((n + 1) * n)

    # Vectorized angular synthesis:
    # - _pi/tau have shape (nmax, n_theta)
    # - an/bn have shape (n_lambda, nmax)
    # The matrix products below evaluate all theta and wavelength combinations
    # in one shot, replacing the previous Python loop over angles.
    weighted_pi = scale[:, None] * _pi
    weighted_tau = scale[:, None] * tau
    S1 = weighted_pi.T @ an.T + weighted_tau.T @ bn.T
    S2 = weighted_tau.T @ an.T + weighted_pi.T @ bn.T

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
    S34 =1*2*(s2.conj()*s1 - s2*s1.conj())

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
    
    # Get scattering amplitude elements S1 and S2
    s1, s2 = scatter_amplitude(wavelength, Nh, Np, D, 
                               theta=theta, 
                               nmax=nmax, 
                               an=an,
                               bn=bn,
                               check_inputs = False)

    # Scale factor
    k_host = 2 * _pi * Nh.real / wavelength
    scale_factor = _np.pi * (k_host * D_shells[-1] / 2.0) ** 2

    # Compute phase function
    phase_fun = 1/scale_factor*(_np.abs(s1)**2 + _np.abs(s2)**2)/2

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
                                index=_pd.Index(_np.degrees(theta), 
                                                name='Theta (deg)'), 
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

    This implementation supports two integration backends over mu = cos(theta):
    - ``gauss``: Gauss-Legendre quadrature.
    - ``trapz``: trapezoidal rule in mu-space.
    Integration method selection is centralized in one internal router.

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
    is_radians = bool(theta_max <= (_np.pi + atol_rad) and theta_min >= -atol_rad)

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

def _mono_percus_yevick(fv, q, D):
    """
    Compute the Percus-Yevick structure factor S(q) for monodispersed 
    hard-sphere systems.

    References: Kinning, D. J., & Thomas, E. L. (1984). 
                Hard-Sphere Interactions between Spherical Domains in Diblock Copolymers. 
                Macromolecules, 17(9), 1712–1718.

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
    
    R = D / 2
    x = 2 * q * R  # Scattering variable as defined by Kinning & Thomas

    # Coefficients from Eq. (17)
    α = (1 + 2 * fv)**2 / (1 - fv)**4
    β = -6 * fv * (1 + fv / 2)**2 / (1 - fv)**4
    γ = 0.5 * fv * (1 + 2 * fv)**2 / (1 - fv)**4

    # G(A) from Eq. (21)
    term1 = α / x**2 * (_np.sin(x) - x * _np.cos(x))
    term2 = β / x**3 * (2 * x * _np.sin(x) + (2 - x**2) * _np.cos(x) - 2)
    term3 = γ / x**5 * (-x**4 * _np.cos(x) +
                        4 * ((3 * x**2 - 6) * _np.cos(x) +
                                (x**3 - 6 * x) * _np.sin(x) + 6))
    G_kt = term1 + term2 + term3

    # Structure factor from Eq. (20)
    S_q = 1 / (1 + 24 * fv * G_kt / x)
    return S_q

def _poly_percus_yevick(fv, qq, D, nD):
    """
    Compute the Percus-Yevick structure factor S(q) for polydisperse 
    hard-sphere systems.

    References: Botet, R., Kwok, R., & Cabane, B. (2020). 
                Percus–Yevick structure factors made simple. 
                Journal of Applied Crystallography, 53(6), 1526–1534.

    Parameters:
    -----------
    fv : float
        Volume fraction (phi) of the spheres.
    qq : ndarray
        Magnitude of the scattering vector.
    D : ndarray
        Diameter of the spheres
    nD : _np.ndarray or None
        Probability distribution over D (same length as D). If None, assumes monodisperse.

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

    R = D / 2

    # if fv > 0.5, compute structure factor for voids
    # "complementary PY hard-sphere approach"
    if fv > 0.5:
        R = (1 - fv)/fv*R
        fv = 1 - fv

    # Vectorized over wavelength and theta dimensions to avoid Python loops.
    # qq shape: (n_wavelengths, n_theta)
    x = qq[:, :, None] * R[None, None, :]  # (n_wavelengths, n_theta, n_bins)

    # Psi is an auxiliary prefactor: psi = 3*phi / (1 - phi)
    psi = 3 * fv / (1 - fv)

    nD_w = nD[None, None, :]
    den_x3 = _trapz((x**3) * nD_w, R, axis=2)

    # Trigonometric building blocks for structure factor (Botet et al., Eqs. 8–13)
    Fcs = _np.cos(x) + x * _np.sin(x)  # cos(x) + x·sin(x)
    Fsc = _np.sin(x) - x * _np.cos(x)  # sin(x) - x·cos(x)

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

    # Auxiliary variables for S(q)
    denom = d**2 + e**2
    X = 1 + b + (2 * e * f * g + d * (f**2 - g**2)) / denom
    Y = c + (2 * d * f * g - e * (f**2 - g**2)) / denom

    # Final expression of S(q) (Eq. 4)
    S_q = (Y / c) / (X**2 + Y**2)
    if squeeze_out:
        return S_q[0]
    return S_q

@_hide_signature
def structure_factor_PY(wavelength: _Union[float, _np.ndarray], 
                        Nh: _Union[float, _np.ndarray], 
                        D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]], 
                        fv: float,
                        *,
                        theta: _Union[float, _np.ndarray] = None,  
                        size_dist: _np.ndarray = None, 
                        check_inputs: bool = True):
    """
    Compute the Percus-Yevick structure factor S(q) for hard-sphere systems,
    for both monodisperse and polydisperse cases.

    Parameters:
    -----------
    wavelength : ndarray or float
        Wavelengtgh (microns)
    
    Nh : ndarray or float
        Complex refractive index of host. If ndarray, len(Nh) == len(wavelength)
    
    D : float or ndarray
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
    
    fv : float
        Volume fraction (phi) of the spheres.
    
    theta : float or ndarray (optional)
        Scattering angle (radians). Default None
    
    size_dist : ndarray (optional)
        Diameter density distribution. len(size_dist) == len(D). If None, assumes monodisperse.
        Default None
    
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
    
    if check_inputs:
        wavelength, Nh, _, D, size_dist = _check_mie_inputs(wavelength, Nh, D = D, 
                                                     size_dist = size_dist)

    # compute scattering vector (q = 2k0*sin(theta/2))
    k0 = 2*_np.pi*Nh.real/wavelength
    q = _np.outer(2*k0, _np.sin(theta/2))

    q[q < 0.1] = 0.1  # Found overflow for q < 0.1
    
    if size_dist is None:
        S_q = _mono_percus_yevick(fv, q, D[-1]).T

    else:
        S_q = _poly_percus_yevick(fv, q, D[-1], size_dist).T

    return S_q

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
                        as_ndarray: bool = False,
                        check_inputs: bool = True,
                        effective_medium: bool = False,
                        dependent_scatt: bool = False):
    """
    Calculate the scattering phase function for multiple hard-spheres under unpolarized light. 
    The intensity is normalized such that the integral is equal to qsca

    Parameters:
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
        phase_fun: the scattering phase function (as pd.DataFrame or ndarray)
    """
    # Input checks
    if check_inputs:
            wavelength, Nh, Np, D, size_dist = _check_mie_inputs(wavelength, Nh, Np, D,
                                                         size_dist=size_dist)
    
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
            an_bin, bn_bin = coeff_cache[0]
        phase_fun = _phase_function_single(wavelength, Nh, Np, D,
                                         theta=theta, 
                                         nmax=nmax, 
                                         an=an_bin,
                                         bn=bn_bin,
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
                an_bin, bn_bin = coeff_cache[i]
            # For each diameter, compute phase function
            phase_fun += size_dist[i] * Ac[i] * _phase_function_single(wavelength, Nh, Np, Di,
                                                                     theta=theta, 
                                                                     nmax=nmax, 
                                                                     an=an_bin,
                                                                     bn=bn_bin,
                                                                     as_ndarray=True, 
                                                                     check_inputs=False)
        
        # Normalize by average cross-sectional area
        phase_fun /= _np.sum(size_dist * Ac)

    if dependent_scatt:
        # Get structure factor
        S_q = structure_factor_PY(wavelength, Nh, D, fv, 
                                theta=theta,
                                size_dist=size_dist, 
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
        Whether to compute an effective host refractive index via Bruggeman EMT (default: True)

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

    Returns
    -------
    cabs_av : _np.ndarray, shape (nλ,)
        Size-averaged scattering cross section per particle [µm²].
    
    csca_av : _np.ndarray, shape (nλ,)
        Size-averaged absorption cross section per particle [µm²].
    
    g_av : _np.ndarray, shape (nλ,)
        Size-averaged asymmetry parameter (⟨cosθ⟩).

    phase_fun_df : pd.DataFrame or None
        Scattering phase function (if `phase_function=True`), with index=θ° and columns=λ.
        Otherwise, None.
    """
    # Input checks
    if check_inputs:
            wavelength, Nh, Np, D, size_dist = _check_mie_inputs(wavelength, Nh, Np, D,
                                                                size_dist=size_dist)

    # Use compact Gauss-Legendre angles by default for phase-dependent paths.
    # Endpoints (0 and pi) are added explicitly so downstream coverage checks
    # remain robust for phase-function integration.

    # asses if fv is within 0 and 1
    if not (0 <= fv < 1):
        raise ValueError("Filling fraction fv must be in the range [0, 1).")
    
    # ---------- Effective medium for host (if your convention is to dress Nh) ----------
    if effective_medium and fv > 0:
        Nh = _effective_host(fv, Np, Nh, D, size_dist,
                            emt_multilayer_fn=_emt_multilayer_sphere,
                            emt_brugg_fn=_emt_brugg,
                            )

    Ac = _np.pi*(D[-1]/2)**2                                  # cross-sectional area of each sphere
    n_bins = 1 if size_dist is None else len(size_dist)       # number of size bins
    p = _np.asarray([1.0]) if size_dist is None else size_dist  # probability for each size bin

    # ---------- Absorption: average q_abs * area ----------
    cabs_av = _np.zeros_like(wavelength, dtype=float)
    csca_av = _np.zeros_like(wavelength, dtype=float)
    gcos_av = _np.zeros_like(wavelength, dtype=float)
    # Cache an/bn per size bin so the optional phase-function path can reuse
    # coefficients instead of recomputing them inside phase_scatt_ensemble.
    coeff_cache = []
    for i in range(n_bins):
        Di = [d[i] for d in D]           # diameter of each layer for current size bin

        # mie.scatter_efficiency must return arrays shaped (nλ,)
        qabs, qsca, gcos, an, bn = scatter_efficiency(wavelength, Nh, Np, Di,
                                                        nmax=nmax,
                                                        return_coeffs=True,
                                                        check_inputs=False,
                                                        )
        coeff_cache.append((an, bn))

        # sanitize any tiny negative due to numerics
        cabs_av += p[i] * qabs * Ac[i]
        csca_av += p[i] * qsca * Ac[i]
        gcos_av += p[i] * qsca * gcos * Ac[i]  # weighted by scattering

    with _np.errstate(divide='ignore', invalid='ignore'):
        gcos_av = _np.divide(
            gcos_av,
            csca_av,
            out=_np.zeros_like(gcos_av),
            where=~_np.isclose(csca_av, 0.0),
        )

    if not phase_function and not dependent_scatt:
        return cabs_av, csca_av, gcos_av, None
    
    # create angular grid for phase function integration
    theta = _np.linspace(0.0, _np.pi, n_theta)  # radians; preserved as radians in phase_scatt_ensemble

    # ---------- Scattering and g: via dense phase function integration ----------
    # phase_scatt_ensemble returns a DataFrame with index=theta [rad] and columns=lambda.
    phase_fun_df = phase_scatt_ensemble(wavelength, theta, Nh, Np, D, fv,
                                        size_dist=size_dist, 
                                        nmax=nmax,
                                        coeff_cache=coeff_cache,
                                        as_ndarray=False, 
                                        effective_medium=False,
                                        check_inputs=False, 
                                        dependent_scatt=dependent_scatt)
    
    # Re-integrate ensemble phase function to recover Qsca and g consistently
    # with the selected quadrature backend.
    qsca_av, gcos_av = scatter_from_phase_function(phase_fun_df)

    # Convert Q_sca (efficiency) to cross section via weighted area ⟨A⟩ = Σ p_i A_i
    A_mean = float(_np.sum(p * Ac))
    csca_av = qsca_av * A_mean

    if not phase_function:
        return cabs_av, csca_av, gcos_av, None

    return cabs_av, csca_av, gcos_av, phase_fun_df

