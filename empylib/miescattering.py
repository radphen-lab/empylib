# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:38:11 2021

@author: PanxoPanza
"""
import numpy as _np
from numpy import pi as _pi, exp as _exp, conj as _conj, imag as _imag, real as _real, sqrt as _sqrt
from scipy.special import jv as _jv, yv as _yv
from .nklib import emt_brugg as _emt_brugg, emt_multilayer_sphere as _emt_multilayer_sphere
from .utils import _as_1d_array, _check_mie_inputs, _check_theta, _hide_signature
import pandas as _pd
from typing import Union as _Union, Optional as _Optional, List as _List

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
    
    # if x is scalar, transform variable to numpy array of dim 1
    if _np.isscalar(x): x = _np.array([x])
    
    n = _np.array(range(nmax))
    
    # Get Dn(x) by downwards recurrence
    Dnx = _np.zeros((len(x),nmx),dtype=_np.complex128)
    for i in reversed(range(1, nmx)):
        # define D_(i+1) (x)
        # if i == nmx-1 : Dip1 = _np.zeros(len(x))
        # else :          Dip1 = Dnx[:,i+1]
        
        Dnx[:,i-1] = (i+1)/x - 1/(Dnx[:,i] + (i+1)/x)
        
    # Get Gn(x) by upwards recurrence
    Gnx = _np.zeros((len(x),nmx),dtype=_np.complex128)
    G0x = 1j*_np.ones_like(x)
    i = 0
    Gnx[:,i] = 1/((i+1)/x - G0x) - (i+1)/x
    for i in range(1, nmx):
        # define G_(i-1) (x)
        # if i == 0 : Gim1x = 1j*_np.ones(len(x))
        # else : Gim1x = Gnx[:,i-1] 
        
        Gnx[:,i] = 1/((i+1)/x - Gnx[:,i-1]) - (i+1)/x
    
    # Get Rn(x) by upwards recurrence
    Rnx = _np.zeros((len(x),len(n)),dtype=_np.complex128) 
    for ix in range(len(x)):
        
        # note that 0.5*(1 - _exp(-2j*x)) = 0 if x = _pi*n
        # I added this clause for those cases
        if _imag(x[ix]) == 0 and _np.mod(_real(x[ix]),_pi) == 0:
            nu = (n + 1) + 0.5
            py =  _sqrt(0.5*_pi*x[ix])*_jv(nu,x[ix])
            chy = _sqrt(0.5*_pi*x[ix])*_yv(nu,x[ix])
            gsy = py + 1j*chy
            Rnx[ix,:] = py/gsy
        
        # otherwise just do normal upward recursion
        else :            
            for i in range(nmax):
                if i == 0 : Rim1x = 0.5*(1 - _exp(-2j*x[ix]))
                else :      Rim1x = Rnx[ix,i-1]
                
                Rnx[ix,i] = Rim1x*(Gnx[ix,i] + (i + 1)/x[ix])/  \
                                  (Dnx[ix,i] + (i + 1)/x[ix]) 
                                  
    return Dnx[:,n], Gnx[:,n], Rnx[:,n]

def _recursive_ab(m, n, Dn, Gn, Rn, Dn1, Gn1, Rn1) :
    i = Dn.shape[0]
    if i == 0:
        an = _np.zeros(n)
        bn = _np.zeros(n)
    else:
        # get an^i and bn^i
        (an, bn) = _recursive_ab(m[:i],n,
                                Dn[:i-1,:], Gn[:i-1,:], Rn[:i-1,:],
                                Dn1[:i-1,:],Gn1[:i-1,:],Rn1[:i-1,:])
        
        # get Un(mi*kri), Vn(mi, kri)
        Un = (Rn[i-1,:]*Dn[i-1,:] - an*Gn[i-1,:])/ \
                (Rn[i-1,:] - an + 1E-10)
        Vn = (Rn[i-1,:]*Dn[i-1,:] - bn*Gn[i-1,:])/ \
                (Rn[i-1,:] - bn + 1E-10)
        
        # get an^(i+1), bn^(i+1) by recursion formula
        an = Rn1[i-1,:]*(m[i]/m[i-1]*Un - Dn1[i-1,:])/ \
                        (m[i]/m[i-1]*Un - Gn1[i-1,:])
                      
        bn = Rn1[i-1,:]*(Vn - m[i]/m[i-1]*Dn1[i-1,:])/ \
                        (Vn - m[i]/m[i-1]*Gn1[i-1,:])

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
    x = _np.asarray(x).ravel()
    assert len(x) == len(m)
    
    ka = x[-1] # size parameter of outer layer

    # define nmax according to B.R Johnson (1996)
    if nmax is None :
        nmax = int(_np.round(_np.abs(ka) + 4*_np.abs(ka)**(1/3) + 2))
    
    #----------------------------------------------------------------------
    #       Computing an and bn (main part of this code)
    #----------------------------------------------------------------------
    
    mix = m*x               # Ni*k*ri
    mi1 = _np.append(m,1)
    mi1x = mi1[1:]*x        # Ni+1*k*ri
    
    # Computation of Dn(z), Gn(z) and Rn(z)
    nmx = int(_np.round(max(nmax, max(abs(m*x))) + 16))
    
    # Get Dn(mi*x), Gn(mi*x), Rn(mi*x) 
    Dn, Gn, Rn = _log_RicattiBessel(mix,nmax,nmx)
    
    # Get Dn(mi+1*x), Gn(mi+1*x), Rn(mi+1*x)
    Dn1, Gn1, Rn1 = _log_RicattiBessel(mi1x,nmax,nmx)
    
    # Get an and bn
    an, bn = _recursive_ab(mi1, nmax, Dn, Gn, Rn, 
                                     Dn1, Gn1, Rn1)
    
    # ---------------------------------------------------------------------
    #       computing secondary paramters
    # ---------------------------------------------------------------------
    # Get Bessel-Ricatti functions and derivatives for last shell layer
    n = _np.array(range(1,nmax+1))
    nu = n+0.5
    phi = _np.sqrt(0.5*_pi*ka)*_jv(nu,ka) # phi(n,ka)
    chi = _np.sqrt(0.5*_pi*ka)*_yv(nu,ka) # chi(n,ka)
    xi  = phi + 1j*chi                    # xi(n,ka)
    
    return an.reshape(-1), bn.reshape(-1), phi, Dn1[-1,:].reshape(-1), xi, Gn1[-1,:].reshape(-1)

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
    '''
    assert len(x) == len(m)
    
    # determine nmax 
    y = x[-1] # size parameter of outer layer

    if nmax is None :
        # define nmax according to B.R Johnson (1996)
        nmax = int(_np.round(_np.abs(y) + 4*_np.abs(y)**(1/3) + 2))

    #------------------------------------------------------------------
    # Get mie coefficient and other parameters of interest
    #------------------------------------------------------------------
    (an, bn, py, Dy, xy, Gy) = _get_coated_coefficients(m,x,nmax)

    if _imag(y) > 1E-8 :
        imy = 2*_imag(y)
        ft = imy**2/(1 + (imy - 1)*_exp(imy))
    else:
        ft = 2
    
    # arranging pre-computing constants
    n = _np.array(range(1,nmax+1))
    
    #------------------------------------------------------------------
    # Extinction efficiency
    #------------------------------------------------------------------
    en = (2*n+1)*_imag((- 2j*py*_conj(py)*_imag(Dy)         \
                       + _conj(an)*_conj(xy)*py*Dy         \
                       - _conj(bn)*_conj(xy)*py*_conj(Gy)   \
                       + an*xy*_conj(py)*Gy               \
                       - bn*xy*_conj(py)*_conj(Dy))        \
                       /y)
    q = _np.sum(en)
    Qext = _real(1/_real(y)*ft*q)    
    
    #------------------------------------------------------------------
    # Scattering efficiency
    #------------------------------------------------------------------
    en = (2*n+1)*_imag((+ _np.abs(an*xy)**2*Gy                \
                       - _np.abs(bn*xy)**2*_conj(Gy)         \
                       )/y)
    q = _np.sum(en)
    Qsca = _real(1/_real(y)*ft*q)
    
    #------------------------------------------------------------------
    # Asymmetry parameter
    #------------------------------------------------------------------
    anp1 = _np.zeros(nmax,dtype=_np.complex128)
    bnp1 = _np.zeros(nmax,dtype=_np.complex128)
    anp1[:nmax-1] = an[1:] # a(n+1) coefficient
    bnp1[:nmax-1] = bn[1:] # a(n+1) coefficient
    
    asy1 = n*(n + 2)/(n + 1)*(an*_conj(anp1)+ bn*_conj(bnp1)) \
         + (2*n + 1)/(n*(n + 1))*_real(an*_conj(bn))
    
    asy2 = (2*n+1)*(an*_conj(an) + bn*_conj(bn))
    Asym = _real(2*_np.sum(asy1)/_np.sum(asy2))
    
    #------------------------------------------------------------------
    # Backward scattering (not valid for absorbing host media)
    #------------------------------------------------------------------
    f = (2*n+1)*(-1)**n*(an - bn)
    q = _np.sum(f)
    Qb = _real(q*_conj(q)/y**2)
    
    #------------------------------------------------------------------
    # Forward scattering (not valid for absorbing host media)
    #------------------------------------------------------------------
    f = (2*n+1)*(an + bn)
    q = _np.sum(f)
    Qf = _real(q*_conj(q)/y**2)
    
    #------------------------------------------------------------------
    # Condition outputs to avoid unphysical results
    #------------------------------------------------------------------
    if Qsca < 0: Qsca = 0
    if Qext < Qsca: Qext = Qsca
    if Asym < -1: Asym = -1
    if Asym > +1: Asym = +1

    return Qext, Qsca, Asym, Qb, Qf

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

    m = (Np / Nh.real).transpose()
    R = D_shells / 2.0
    kh = 2 * _pi * Nh.real / wavelength
    x = _np.outer(kh, R)
    
    # Preallocate outputs
    qext = _np.zeros_like(wavelength, dtype=float)
    qsca = _np.zeros_like(wavelength, dtype=float)
    gcos = _np.zeros_like(wavelength, dtype=float)
    for i in range(len(wavelength)):
        qext[i], qsca[i], gcos[i], *_ = _cross_section_at_lam(m[i, :], x[i, :], nmax)

    # outputs: qabs, qsca, gcos
    qabs = qext - qsca
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

    m = (Np / Nh.real).transpose()
    R = D_shells / 2.0
    kh = 2 * _pi * Nh.real / wavelength
    x = _np.outer(kh, R)

    # determine nmax 
    if nmax is None :
        y = max(x[-1,:]) # largest size parameter of outer layer
        # define nmax according to B.R Johnson (1996)
        nmax = int(_np.round(_np.abs(y) + 4*_np.abs(y)**(1/3) + 2))

    # Preallocate outputs
    an = _np.zeros((len(wavelength), nmax), dtype=complex)
    bn = _np.zeros((len(wavelength), nmax), dtype=complex)
    for i in range(len(wavelength)):
        an[i,:], bn[i, :], *_ = _get_coated_coefficients(m[i, :], x[i, :], nmax)
    
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

    # Extract mie scattering coefficients
    an, bn = scatter_coefficients(wavelength, Nh, Np, D, 
                                nmax=nmax, 
                                check_inputs=False)
    nmax = an.shape[1]

    # get _pi and tau angular functions
    _pi, tau = _pi_tau_1n(theta, nmax)

    # set scale for summation
    n = _np.arange(1, nmax + 1)
    scale = (2 * n + 1) / ((n + 1) * n)

    mu = _np.cos(theta)

    # compute S1 and S2
    S1 = _np.zeros((len(mu), len(wavelength)), dtype=_np.complex128)
    S2 = _np.zeros((len(mu), len(wavelength)), dtype=_np.complex128)
    for k in range(len(mu)):
        S1[k] = _np.dot(scale* _pi[:,k],an.T) + _np.dot(scale*tau[:,k],bn.T)
        S2[k] = _np.dot(scale*tau[:,k],an.T) + _np.dot(scale* _pi[:,k],bn.T)

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
        phase_fun: the scattering phase function (as pd.DataFrame or ndarray)
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
                               check_inputs = False)

    # Scale factor
    x = _np.pi * Nh.real * D_shells[-1] / wavelength
    scale_factor = _np.pi*x**2

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
                                n_theta: int = 100, 
                                atol_deg: float = 1.0):
    """
    Compute Qsca and <cos theta> from a DataFrame whose rows are labeled
    with scattering angles in degrees and columns with wavelengths.

    This version (a) verifies coverage of [0°, 180°] and (b) interpolates the
    phase function onto a regular theta grid in [0°, 180°] before integrating,
    while preserving the original integration in mu = cos(theta).

    Parameters
    ----------
    phase_fun : pd.DataFrame
        Phase function. Row index must be theta in degrees (not necessarily uniform).
        Columns correspond to different wavelengths.
    
    n_theta : int, optional
        Number of points for the interpolation grid in [0°, 180°]. Default 181 (~0.25°).
    
    atol_deg : float, optional
        Tolerance (in degrees) to accept coverage near 0° and 180°. Default 1.0°.

    Returns
    -------
    qsca : ndarray
        Scattering efficiency for each column.
    
    gcos : ndarray
        Asymmetry parameter for each column.
    """
    if not isinstance(phase_fun, _pd.DataFrame):
        raise TypeError("phase_fun must be a pandas DataFrame (angles as index in degrees).")

    # ---- 1) Sort and basic checks ----
    pf = phase_fun.sort_index().copy()
    # ensure numeric index (degrees)
    try:
        theta_all = _np.asarray(pf.index, dtype=float)
    except Exception as e:
        raise TypeError("Row index must be numeric degrees (float).") from e

    if theta_all.size < 2:
        raise ValueError("Theta index must contain at least two samples.")

    # ---- 2) Verify coverage of [0°, 180°] (allow larger theta; just require at least this span) ----
    tmin, tmax = float(_np.nanmin(theta_all)), float(_np.nanmax(theta_all))
    if tmin > 0.0 + atol_deg or tmax < 180.0 - atol_deg:
        raise ValueError(
            f"Theta index must cover at least [0°, 180°] within ±{atol_deg}°. "
            f"Got range [{tmin:.3f}°, {tmax:.3f}°]."
        )

    # ---- 3) Clip to [0°, 180°] (keep only the needed range for integration) ----
    pf = pf.loc[(pf.index >= 0.0 - atol_deg) & (pf.index <= 180.0 + atol_deg)].copy()
    # Consolidate duplicates by averaging
    pf.index = _np.asarray(pf.index, dtype=float)
    pf = pf.groupby(level=0).mean().sort_index()

    # Re-check span after clipping/cleaning
    theta_clipped = pf.index.to_numpy()
    if theta_clipped[0] > 0.0 + atol_deg or theta_clipped[-1] < 180.0 - atol_deg:
        raise ValueError("After clipping/cleaning, theta does not span [0°, 180°] within tolerance.")

    # ---- 4) Interpolate onto a regular theta grid in [0°, 180°] ----
    if n_theta < 2:
        raise ValueError("n_theta must be >= 2.")
    theta_eval_deg = _np.linspace(0.0, 180.0, n_theta)              # degrees
    theta_eval_rad = _np.radians(theta_eval_deg)                     # radians (for cos)
    ncols = pf.shape[1]
    P_eval = _np.empty((n_theta, ncols), dtype=float)

    x_src = theta_clipped
    for j, col in enumerate(pf.columns):
        y_src = pf[col].to_numpy()
        mask = _np.isfinite(y_src)
        xj = x_src[mask]
        yj = y_src[mask]
        if xj.size < 2:
            raise ValueError(f"Column '{col}': not enough finite samples to interpolate.")
        if xj.min() > 0.0 + atol_deg or xj.max() < 180.0 - atol_deg:
            raise ValueError(f"Column '{col}': finite data must span [0°, 180°] within tolerance.")
        # Linear interpolation within the convex hull
        P_eval[:, j] = _np.interp(theta_eval_deg, xj, yj)

    # ---- 5) Change variable to mu = cos(theta) and integrate over mu ----
    mu = _np.cos(theta_eval_rad)              # mu is descending from +1 to -1 as theta increases
    order = _np.argsort(mu)                   # sort ascending mu (-1 .. 1)
    mu_sorted = mu[order]
    p_sorted = P_eval[order, :]               # reorder rows to match ascending mu

    # Compute Qsca and g using trapezoidal rule in mu
    qsca = 2.0 * _np.pi * _np.trapz(p_sorted, mu_sorted, axis=0)
    with _np.errstate(divide='ignore', invalid='ignore'):
        gcos = (2.0 * _np.pi * _np.trapz((mu_sorted[:, None] * p_sorted), mu_sorted, axis=0)) / qsca

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

    R = D / 2

    # Weighted average over size distribution
    average = lambda f: _np.trapz(f * nD, R, axis = 1)  

    # if fv > 0.5, compute structure factor for voids
    # "complementary PY hard-sphere approach"
    if fv > 0.5:
        R = (1 - fv)/fv*R
        fv = 1 - fv

    S_q = _np.zeros_like(qq)
    for i in range(qq.shape[0]):
        q = _np.meshgrid(R, qq[i,:])[1]
        
        x = q * R  # Scattering vector scaled by radius
        
        # Psi is an auxiliary prefactor: psi = 3*phi / (1 - phi)
        psi = 3 * fv / (1 - fv)
    
        # Trigonometric building blocks for structure factor (Botet et al., Eqs. 8–13)
        Fcs = _np.cos(x) + x * _np.sin(x)  # cos(x) + x·sin(x)
        Fsc = _np.sin(x) - x * _np.cos(x)  # sin(x) - x·cos(x)
    
        # Botet et al. expressions for b, c, d, e, f, g
        b = psi * average(Fcs * Fsc) / average(x**3)
        c = psi * average(Fsc**2) / average(x**3)
        d = 1 + psi * average(x**2 * _np.sin(x) * _np.cos(x)) / average(x**3)
        e = psi * average(x**2 * _np.sin(x)**2) / average(x**3)
        f = psi * average(x * _np.sin(x) * Fsc) / average(x**3)
        g = - psi * average(x * _np.cos(x) * Fsc) / average(x**3)

        
        # Auxiliary variables for S(q)
        denom = d**2 + e**2
        X = 1 + b + (2 * e * f * g + d * (f**2 - g**2)) / denom
        Y = c + (2 * d * f * g - e * (f**2 - g**2)) / denom
    
        # Final expression of S(q) (Eq. 4)
        S_q[i,:] = (Y / c) / (X**2 + Y**2)
        
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
                        Nh: _Union[float, _np.ndarray],
                        Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
                        D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],
                        fv: float = 0.0,
                        *, 
                        size_dist: _np.ndarray = None, 
                        theta: _Union[float, _np.ndarray] = None,
                        nmax: int = None, 
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

    N_layers = len(D)                                    # number of layers in the sphere
    if effective_medium and fv > 0:
        # Compute effective refractive index of host using Bruggeman EMT
        D_layers_mean = []
        for i in range(N_layers):
            if size_dist is None:
                # Monodisperse
                D_layers_mean.append(float(_np.asarray(D[i]).ravel()[0]))
            else:
                # Polydisperse
                D_layers_mean.append(_np.average(D[i], axis=0,   # -> float
                                            weights=size_dist))  # size_dist shape (n_bins,)
                                        
        Np_eff = _emt_multilayer_sphere(D_layers_mean, Np, check_inputs=False)
        Nh = _emt_brugg(fv, Np_eff, Nh)

    # Get form factor
    if size_dist is None:
        # Monodisperse
        phase_fun = _phase_function_single(wavelength, Nh, Np, D,
                                         theta=theta, 
                                         nmax=nmax, 
                                         as_ndarray=True, 
                                         check_inputs=False)
    else:
        Ac = _np.pi*(D[-1]/2)**2  # cross-sectional area of each sphere

        # Polydisperse: ensemble average over diameter distribution
        phase_fun = _np.zeros((len(theta), len(wavelength)), dtype=float)
        for i in range(len(size_dist)):
            Di = [d[i] for d in D]  # diameter of each layer for current size bin
            # For each diameter, compute phase function
            phase_fun += size_dist[i] * Ac[i] * _phase_function_single(wavelength, Nh, Np, Di,
                                                                     theta=theta, 
                                                                     nmax=nmax, 
                                                                     as_ndarray=True, 
                                                                     check_inputs=True)
        
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
                                 index=_np.degrees(theta), 
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
    theta: _Union[float, _np.ndarray] = None,
    nmax: int = None, 
    check_inputs: bool = True,
    effective_medium: bool = False,
    dependent_scatt: bool = False,
    phase_function: bool = False
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
    
    D : float, _np.ndarray or list
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        if multilayer sphere, use list of floats (monodisperse) or list of arrays (polydisperse).
    
    fv : float
        Particle volume fraction in (0, 1). Used only to compute an effective medium Nh via
        `nk.emt_brugg(fv, Np, Nh)`.
    
    size_dist : array-like, shape (nD,)
        Number-fraction probabilities for each diameter (Case A). Sum must be 1
        within tolerance; will be renormalized if slightly off.
    
    theta : float or array-like, optional
        Scattering angle(s) in radians for phase function integration (default: None,
    
    nmax : int, optional
        Number of mie scattering coefficients (default: None, automatic).
        
    check_inputs : bool, optional
        Whether to check mie inputs (default: True)    

    effective_medium : bool, optional
        Whether to compute an effective host refractive index via Bruggeman EMT (default: True)

    dependent_scatt : bool, optional
        Whether to include dependent scattering effects via Percus-Yevick structure factor
        (default: False; not recommended for metallic spheres or high fv)    
    
    phase_function : bool, optional
        If True, also return the phase function DataFrame (default: False)    

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

    # checks variable theta
    theta = _check_theta(theta)

    # asses if fv is within 0 and 1
    if not (0 <= fv < 1):
        raise ValueError("Filling fraction fv must be in the range [0, 1).")
    
    # ---------- Effective medium for host (if your convention is to dress Nh) ----------
    N_layers = len(D)                                    # number of layers in the sphere
    if effective_medium and fv > 0:
        # Compute mean diameter of each layer
        D_layers_mean = []
        for i in range(N_layers):
            if size_dist is None:
                D_layers_mean.append(float(_np.asarray(D[i]).ravel()[0]))
            else:
                # Polydisperse
                D_layers_mean.append(_np.average(D[i], axis=0,   # -> float
                                            weights=size_dist))  # size_dist shape (n_bins,)

        # Compute effective refractive index of host using Bruggeman EMT                                   
        Np_eff = _emt_multilayer_sphere(D_layers_mean, Np, check_inputs=False)
        Nh = _emt_brugg(fv, Np_eff, Nh)

    Ac = _np.pi*(D[-1]/2)**2                                  # cross-sectional area of each sphere
    n_bins = 1 if size_dist is None else len(size_dist)       # number of size bins
    p = _np.asarray([1.0]) if size_dist is None else size_dist  # probability for each size bin

    # ---------- Absorption: average q_abs * area ----------
    cabs_av = _np.zeros_like(wavelength, dtype=float)
    csca_av = _np.zeros_like(wavelength, dtype=float)
    gcos_av = _np.zeros_like(wavelength, dtype=float)
    for i in range(n_bins):
        Di = [d[i] for d in D]           # diameter of each layer for current size bin

        # mie.scatter_efficiency must return arrays shaped (nλ,)
        qabs, qsca, gcos = scatter_efficiency(wavelength, Nh, Np, Di, 
                                              nmax = nmax,
                                              check_inputs = False)

        # sanitize any tiny negative due to numerics
        cabs_av += p[i] * qabs * Ac[i]
        csca_av += p[i] * qsca * Ac[i]
        gcos_av += p[i] * gcos * qsca * Ac[i]  # weighted by scattering

    gcos_av /= csca_av  # normalize by total scattering

    if not phase_function and not dependent_scatt:
        return cabs_av, csca_av, gcos_av, None

    # ---------- Scattering and g: via dense phase function integration ----------
    # phase_scatt_ensemble should return a DataFrame with index=θ° and columns=λ (your earlier design)
    phase_fun_df = phase_scatt_ensemble(wavelength, Nh, Np, D, fv,
                                        size_dist=size_dist, 
                                        theta=theta,
                                        nmax=nmax,
                                        as_ndarray=False, 
                                        effective_medium=False,
                                        check_inputs=False, 
                                        dependent_scatt=dependent_scatt)

    # Compute Q_sca and g from differential efficiency
    qsca_av, gcos_av = scatter_from_phase_function(phase_fun_df)

    # Convert Q_sca (efficiency) to cross section via weighted area ⟨A⟩ = Σ p_i A_i
    A_mean = float(_np.sum(p * Ac))
    csca_av = qsca_av * A_mean

    if not phase_function:
        return cabs_av, csca_av, gcos_av, None

    return cabs_av, csca_av, gcos_av, phase_fun_df

