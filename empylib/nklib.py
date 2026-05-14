# -*- coding: utf-8 -*-
"""
Library of tabulated refractive index

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import numpy as _np 
import pandas as _pd
from scipy.integrate import quad as _quad
from scipy.optimize import least_squares as _least_squares
from typing import Callable as _Callable # used to check callable variables
from pathlib import Path as _Path
# import refidx as ri
from .utils import convert_units as _convert_units, _check_mie_inputs, _warn_extrapolation, _as_1d_array
from typing import List as _List, Union as _Union
import yaml as _yaml
import requests as _requests
from io import StringIO as _StringIO
import inspect as _inspect
import re as _re

__all__ = ('get_nkfile', 'get_ri_info', 
           'lorentz', 'drude', 'tauc_lorentz', 'gaussian', 'multi_oscillator', 'fit_to_oscillator',
           'interactive_oscillator_guess',
           'emt_multilayer_sphere', 'emt_brugg', 'eps_real_kkr',
           'SiO2', 'Silica', 'CaCO3', 'BaSO4', 'BaF2', 'TiO2',
           'BiVO4_mono_a', 'BiVO4_mono_b', 'BiVO4_mono_c', 'BiVO4', 'Cu2O', 'ZnO',
            'MgO', 'Al2O3', 'ZnS', 'GSTa', 'GSTc', 'VO2M', 'VO2R', 'VO2',
            'Si', 'gold', 'silver', 'Cu', 'Al', 'Mg',
            'HDPE', 'PDMS', 'PMMA', 'PVDF', 'H2O')

_OSCILLATOR_BASE_MODELS = {
    'tauc-lorentz': 'tauc_lorentz',
    'gaussian': 'gaussian',
    'lorentz': 'lorentz',
    'drude': 'drude',
}

_OSCILLATOR_DEFAULT_BOUNDS = {
    'drude': {'epsinf': (0, 10), 'wp': (1E-5, 10), 'gamma': (1E-5, 10)},
    'lorentz': {'epsinf': (0, 10), 'wp': (1E-5, 10), 'wn': (1E-2, 10), 'gamma': (1E-5, 10)},
    'tauc-lorentz': {'A':  (1E-5, 100), 'C':  (1E-5, 10), 'E0': (0, 10), 'Eg': (0, 10)},
    'gaussian': {'A':  (1E-5, 100), 'Br':  (1E-5, 100), 'E0': (0, 10)}
}

def blend_model(wavelength, nk_df, nk_model, blend_low=None, blend_high=None):
    '''
    Blend tabulated nk data with a model outside the data range to smooth transition.
    Parameters
    ----------
    wavelength : ndarray
        Wavelengths to interpolate (um).
    nk_df : DataFrame
        Tabulated nk data.
    nk_model : ndarray
        Model values outside the data range.
    blend_low : float, optional
        Lower blending window (um).
    blend_high : float, optional
        Upper blending window (um).
        
    Returns
    -------
    nk_out : ndarray
        Blended complex refractive index
    '''

    # get inside index based on zero values of nk_interp
    lam_min, lam_max = float(nk_df.index[0]), float(nk_df.index[-1])
    inside = (wavelength >= lam_min) & (wavelength <= lam_max)

    # interpolate nk data
    nk_interp = _np.interp(wavelength, nk_df.index, nk_df['n'] + 1j*nk_df['k'])
    nk_out = _np.empty_like(nk_interp, dtype=complex)
    nk_out[~inside] = nk_model[~inside]   # outside data range: model
    nk_out[inside] = nk_interp[inside]    # inside data range: interpolated data
    
    if blend_low is None:
        # set value based on order of magnitude of lam_min
        blend_low = 10**_np.round(_np.log10(lam_min))

    if blend_high is None:
        # set value based on order of magnitude of lam_max
        blend_high = 10**_np.round(_np.log10(lam_max))

    # blend lower end data to smooth transition
    if blend_low > 0:
        # Blend in [wl_min, wl_min+blend_window]
        bw = float(blend_low)

        # lower edge
        low = (wavelength >= lam_min) & (wavelength <= lam_min + bw)
        if _np.any(low):
            t = (wavelength[low] - lam_min) / bw  # 0..1
            # smoothstep
            s = t*t*(3 - 2*t)
            nk_itp = nk_interp[low]
            nk_out[low] = (1 - s)*nk_model[low] + s*nk_itp

    # blend higher end data to smooth transition
    if blend_high > 0:
        bw = float(blend_high)
        # upper edge
        high = (wavelength <= lam_max) & (wavelength >= lam_max - bw)
        if _np.any(high):
            t = (lam_max - wavelength[high]) / bw  # 0..1
            s = t*t*(3 - 2*t)
            nk_itp = nk_interp[high]
            nk_out[high] = (1 - s)*nk_model[high] + s*nk_itp

    return nk_out

def get_nkfile(wavelength, MaterialName=None, get_from_local_path = False, lam_units = 'um', *, 
               extrapolate = 'flat'):
    '''
    Reads a tabulated *.nk file and returns an interpolated
    1D numpy array with the complex refractive index
    
    Parameters
    ----------
    wavelength : ndarray
        Wavelengths to interpolate (um).
    MaterialName : string
        Name of *.nk file
    get_from_local_path : bool
        If True, retrieves nk file from local empylib/nk_files folder. If False, retrieves from working directory.
    lam_units : string
        Units of input wavelength (default 'um'). Options: 'nm', 'um', 'mm', 'm'
    extrapolate : None, string or dict
        Extrapolation method or parameters (default 'flat'). Options: False, 'flat', or dict with oscillator parameters

    Returns
    -------
    N : ndarray
        Interpolated complex refractive index
    data: ndarray
        Original tabulated data from file
    '''
    # retrieve local path
    if get_from_local_path:
        # if function is called locally
        caller_directory = _Path(__file__).parent / 'nk_files'
    else :
        # if function is called from working directory (where the function is called)
        caller_directory = _Path.cwd()
    
    # Construct the full path of the file
    filename = MaterialName + '.nk'
    file_path = caller_directory / filename   
   
    # check if file exist
    assert file_path.exists(), 'File not found'
    
    # read data as dataframe
    nk_df = _pd.read_csv(file_path, comment='#', sep='\s+', header=None, index_col=0)
    
    # check if has n and k data
    assert nk_df.shape[1] == 2, 'wrong file format'

    # label columns and index
    nk_df.columns = ['n', 'k']
    nk_df.index.name = 'lambda'

    # convert wavelength to um
    nk_df.index = _convert_units(nk_df.index, lam_units, to='um')

    return _process_nk_data(wavelength, nk_df, MaterialName, extrapolate)

def ri_info_data(shelf,book,page):
    """
    Reads a YAML file containing 'nk' tabulated optical data from a URL and returns:
    - wavelength: ndarray of wavelengths
    - nk: ndarray of complex refractive indices (n + ik)
    """
    url_root = 'https://refractiveindex.info/database/data/' 
    url = url_root  + shelf + '/'  + book  + '/nk/' + page + '.yml'

    # Download YAML content
    response = _requests.get(url)
    response.raise_for_status()

    # Parse YAML content
    yaml_data = _yaml.safe_load(response.text)

    # Extract tabulated data block
    nk_text = yaml_data['DATA'][0]['data']

    # Read into DataFrame using regex-based separator
    nk_df = _pd.read_csv(_StringIO(nk_text), sep=r'\s+', names=['wavelength', 'n', 'k'])
    nk_df.index = nk_df['wavelength']           # set wavelength as index
    nk_df = nk_df.drop(columns=['wavelength'])  # remove 'wavelength' column

    return nk_df

def get_ri_info(wavelength, shelf, book, page, *, extrapolate='flat'):
    '''
    Extract refractive index from refractiveindex.info database. This code
    uses the refidx package from Bejamin Vial (https://gitlab.com/benvial/refidx)

    Parameters
    ----------
    wavelength : ndarray
        Wavelengths to interpolate (um).
    shelf : string
        Name of the shelf (main, organic, glass, other, 3D)
    book : string
        Material name
    page: string
        Refractive index source   
    extrapolate : bool, string or dict
        Extrapolation method or parameters (default 'flat'). Options: False, 'flat', or dict with oscillator parameters

    Returns
    -------
    N : ndarray
        Interpolated complex refractive index
    data: ndarray
        Original tabulated data from file
    '''
    nk_df = ri_info_data(shelf,book,page)
    MaterialName = book + '_' + page

    return _process_nk_data(wavelength, nk_df, MaterialName, extrapolate)

def _process_nk_data(wavelength, nk_df, MaterialName, extrapolate):
    '''
    Process nk dataframe and interpolate to desired wavelengths.
    
    Parameters
    ----------
    wavelength : ndarray
        Wavelengths to interpolate (um).
    nk_df : DataFrame
        DataFrame containing 'n' and 'k' columns indexed by wavelength.
    MaterialName : string
        Name of the material for labeling purposes.
    extrapolate : bool, string or dict
        Extrapolation method or parameters (default 'flat'). Options: False, 'flat', or dict with oscillator parameters.

    Returns
    -------
    N : ndarray
        Interpolated complex refractive index
    nk_df : DataFrame
        Original tabulated data from file
    '''
    
    # check if wavelength is not ndarray
    wavelength = _as_1d_array(wavelength, name = "wavelength") 

    # create complex refractive index using interpolation form nkfile
    nk_df_complex = nk_df['n'] + 1j*nk_df['k']
    if extrapolate is False:
        N = _np.interp(wavelength, nk_df.index, nk_df_complex, 
                       left = complex(0, 0), right = complex(0, 0))
    
    elif extrapolate == 'flat':
        N = _np.interp(wavelength, nk_df.index, nk_df_complex)
    
    elif isinstance(extrapolate, dict):
        N_model = multi_oscillator(wavelength, extrapolate)
        N = blend_model(wavelength, nk_df, N_model)
    else:
        raise ValueError("Extrapolation method not recognized. Use False, 'flat', or dict with oscillator parameters.") 

    # if N.real or N.imag < 0, make it = 0
    N[N.real<0] = 0                + 1j*N[N.real<0].imag # real part = 0 (keep imag part)
    
    # warning if extrapolated values
    lo, hi = float(nk_df.index[0]), float(nk_df.index[-1])
    _warn_extrapolation(wavelength, lo, hi, label=MaterialName, quantity="refractive index")
    
    # if wavelength was float (orginaly), convert N to a complex value
    return complex(N[0]) if len(N) == 1 else N, nk_df

'''
    --------------------------------------------------------------------
                    dielectric constant models
    --------------------------------------------------------------------
'''
def _split_by_max(arr, threshold):
    """
    Identify and group the indices of elements in the array that are greater than a given threshold.
    Each group contains consecutive indices where the condition is satisfied.

    Parameters:
    ----------
    arr : list or array-like
        The input array to be analyzed.
    threshold : int or float
        The threshold value; only elements greater than this value are considered.

    Returns:
    -------
    list of lists
        A list containing sublists, each with consecutive indices where arr[index] > threshold.
    """
    # Step 1: Find indices where values > 10
    indices = _np.where(_np.array(arr) > threshold)[0]

    # Step 2: Group consecutive indices
    index_list = []
    idx = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            idx.append(indices[i])
        else:
            index_list.append(idx)
            idx = [indices[i]]
    index_list.append(idx)  # Append the last group
    return index_list

def _fix_nk_anomalous(wavelength, n, k):
    '''
    PENDING
    Analyze nk to fix anomalous behaviors. In the case of n, it just makes 
    n = 0 if n < 0. For k, analyze beer-lambert transmittance of a 1 um film. 
    Adjust k that fall T_bl > T_threshold to a very low value

    Parameters
    ------------
    wavelength: ndarray
        Wavelength of tabulated data
    n, k: ndarray
        Tabulated n and k

    Return
    -------

    fixed n and k

    '''
    #---------------------------------------------------------------------
    #                               Fix n 
    #---------------------------------------------------------------------   
    n_new = n.copy()
    n_new[n<0] = 0 + 1j*k[n<0] # real part = 0 (keep imag part)

    #---------------------------------------------------------------------
    #                               Fix k 
    #---------------------------------------------------------------------
    d = 1                                     # Film test thickness (um)
    T_threst = 0.996                          # Transmittance threshold
    a_coef = 4*_np.pi*k/wavelength             # Absorption coefficient of film (1/um)
    T_bl = _np.exp(-a_coef*d)                  # Get Beer-Lambert transmittance
    idx_list = _split_by_max(T_bl, T_threst)  # Find index that pass threshold

    k_new = k.copy()
    for idx in idx_list:

        # Adjust k values to a linear regression with very large slope
        slope = 20                            # slope of the curve
        x0, y0 = _np.log(wavelength[idx[0] ]), _np.log(k[idx[0]])
        b_dw = y0 + slope*x0                  # find y-intersept of downward curve

        x0, y0 = _np.log(wavelength[idx[-1] ]), _np.log(k[idx[-1]])
        b_up = y0 - slope*x0                  # find y-intersept of upward curve

        # find intersection between the two curves
        x_intersect = _np.exp((b_dw - b_up)/(2*slope)) 
        idx_cut = _np.where(wavelength < x_intersect)[0][-1]  # index of intersection

        # create new k values with linear curves
        k_new[idx[0]:idx_cut]  = _np.exp(b_dw - slope*_np.log(wavelength[idx[0]:idx_cut]))
        k_new[idx_cut:idx[-1]] = _np.exp(b_up + slope*_np.log(wavelength[idx_cut:idx[-1]]))

    return n_new + 1j*k_new

def gaussian(wavelength, A,Br,E0):
    '''
    Gaussian oscillator model for dielectric constant based on
    parameters from ellipsometry measurements. The model first calculates
    the imaginary part of epsilon, and the retrieves the real component
    using Krammers-Kronig model.

    Parameters
    ----------
    A   : float
        Absorption amplitude  
    
    Br  : float
        Broadening (eV)
    
    E0  : float
        Oscillator energy (eV)

    wavelength : ndarray
        Wavelengths range (um)

    Returns
    -------
    eps : ndarray (complex)
        Complex dielectric constant
    '''
    # Gauss model as function of E (in eV)
    f = 0.5 / _np.sqrt(_np.log(2))
    E = _as_1d_array(_convert_units(wavelength, 'um', 'eV'), name='energy')

    def _eps_g(Ev):
        return A * _np.exp(-(f * (Ev - E0) / Br)**2) - A * _np.exp(-(f * (Ev + E0) / Br)**2)

    eps_im = _eps_g(E)

    # Vectorized principal-value KK integral on an energy grid.
    e_lo = max(float(_np.min(E)) * 0.2, 1e-6)
    e_hi = max(float(_np.max(E)) * 5.0, E0 + 12.0 * Br)
    n_grid = max(4000, 8 * E.size)
    xi = _np.linspace(e_lo, e_hi, n_grid)
    eps2_xi = _eps_g(xi)

    denom = xi[:, None]**2 - E[None, :]**2

    dxi = xi[1] - xi[0]
    pv_mask = _np.abs(xi[:, None] - E[None, :]) <= 2.0 * dxi
    safe = ~pv_mask

    num = xi[:, None] * eps2_xi[:, None]
    integrand = _np.zeros_like(denom, dtype=float)
    _np.divide(num, denom, out=integrand, where=safe)

    eps_re = (2.0 / _np.pi) * _np.trapz(integrand, x=xi, axis=0)

    eps = eps_re + 1j * eps_im
    n_complex = _np.sqrt(eps)
    return n_complex[0] if n_complex.size == 1 else n_complex

def tauc_lorentz(wavelength, A,C,E0,Eg):
    '''
    Tauc-Lorentz oscillator model for dielectric constant based on
    parameters from ellipsometry measurements.

    Parameters
    ----------
    A   : float
        Oscillator's amplitude  
    
    C  : float
        Broadening of the oscillator(eV)
    
    E0  : float
        Oscillator's resonant energy (eV)

    Eg  : float
        Bandgap (eV)
        
    wavelength : ndarray
        Wavelengths range (um)

    Returns
    -------
    eps : ndarray (complex)
        Complex dielectric constant
    '''
    
    # Tauc-Lorentz imaginary component as function of photon energy E (eV)
    E = _as_1d_array(_convert_units(wavelength, 'um', 'eV'), name='energy')
    eps_im = A * E0 * C * (E - Eg)**2
    eps_im /= E * ((E**2 - E0**2)**2 + C**2 * E**2)
    eps_im[E <= Eg] = 0

    # Closed-form Re(epsilon) from Jellison-Modine (1996).
    # If alpha is near zero (e.g. C ~ 2*E0), use a stable vectorized KK fallback.
    alpha2 = 4.0 * E0**2 - C**2
    if alpha2 > 1e-12:
        alpha = _np.sqrt(alpha2)
        gamma2 = E0**2 - C**2 / 2.0

        a_log = (Eg**2 - E0**2) * E**2 + Eg**2 * C**2 - E0**2 * (E0**2 + 3.0 * Eg**2)
        a_atan = (E**2 - E0**2) * (E0**2 + Eg**2) + Eg**2 * C**2
        zeta4 = (E**2 - gamma2)**2 + (alpha**2) * C**2 / 4.0

        ratio = (E0**2 + Eg**2 + alpha * Eg) / (E0**2 + Eg**2 - alpha * Eg)
        atan_term = _np.pi - _np.arctan((2.0 * Eg + alpha) / C) + _np.arctan((alpha - 2.0 * Eg) / C)
        atan_aux = _np.pi + 2.0 * _np.arctan(2.0 * (gamma2 - Eg**2) / (alpha * C))
        log_ref = _np.sqrt((E0**2 - Eg**2)**2 + Eg**2 * C**2)

        # Avoid log singularity at E=Eg in finite-precision arithmetic.
        EmEg = _np.maximum(_np.abs(E - Eg), 1e-15)
        E_safe = _np.maximum(E, 1e-15)

        eps_re = (
            A * C * a_log / (2.0 * _np.pi * zeta4 * alpha * E0) * _np.log(ratio)
            - A * a_atan / (_np.pi * zeta4 * E0) * atan_term
            + 2.0 * A * E0 * Eg * (E**2 - gamma2) / (_np.pi * zeta4 * alpha) * atan_aux
            - A * E0 * C * (E**2 + Eg**2) / (_np.pi * zeta4 * E_safe) * _np.log(EmEg / (E + Eg))
            + 2.0 * A * E0 * C * Eg / (_np.pi * zeta4) * _np.log(EmEg * (E + Eg) / log_ref)
        )
    else:
        # Degenerate-alpha fallback: principal-value KK integral on a fixed grid.
        e_max = max(float(E.max()) * 5.0, Eg + 50.0 * max(C, 1e-3), E0 + 50.0 * max(C, 1e-3))
        xi = _np.linspace(max(Eg, 1e-6), e_max, 5000)
        eps2_xi = A * E0 * C * (xi - Eg)**2
        eps2_xi /= xi * ((xi**2 - E0**2)**2 + C**2 * xi**2)
        eps2_xi[xi <= Eg] = 0

        denom = xi[:, None]**2 - E[None, :]**2

        dxi = xi[1] - xi[0]
        pv_mask = _np.abs(xi[:, None] - E[None, :]) <= 2.0 * dxi
        safe = ~pv_mask

        num = xi[:, None] * eps2_xi[:, None]
        integrand = _np.zeros_like(denom, dtype=float)
        _np.divide(num, denom, out=integrand, where=safe)

        eps_re = (2.0 / _np.pi) * _np.trapz(integrand, x=xi, axis=0)

    eps = eps_re + 1j * eps_im
    n_complex = _np.sqrt(eps)
    return n_complex[0] if n_complex.size == 1 else n_complex

def lorentz(wavelength, epsinf,wp,wn,gamma):
    '''
    Refractive index from Lorentz model

    Parameters
    ----------
    epsinf : float
        dielectric constant at infinity.
    wp : float
        Plasma frequency, in eV (wp^2 = Nq^2/eps0 m).
    wn : float
        Natural frequency in eV
    gamma : float
        Decay rate in eV
    wavelength : linear _np.array
        wavelength spectrum in um

    Returns
    -------
    complex refractive index

    '''
    from .utils import convert_units
    w = convert_units(wavelength, 'um', 'eV')  # convert from um to eV 
    
    return _np.sqrt(epsinf + wp**2/(wn**2 - w**2 - 1j*gamma*w))

def drude(wavelength, epsinf,wp,gamma):
    '''
    Refractive index from Drude model

    Parameters
    ----------
    epsinf : float
        dielectric constant at infinity.
    wp : float
        Plasma frequency, in eV (wp^2 = Nq^2/eps0 m).
    gamma : float
        Decay rate in eV
    wavelength : linear _np.array
        wavelength spectrum in um

    Returns
    -------
    complex refractive index

    '''
    # define constants
    eV = 1.602176634E-19          # eV to J (conversion)
    hbar = 1.0545718E-34          # J*s (plank's constan)
    
    
    w = 2*_np.pi*3E14/wavelength*hbar/eV  # convert from um to eV 
    
    return _np.sqrt(epsinf - wp**2/(w**2 + 1j*gamma*w))

def _normalize_fixed_params(fixed_params):
    '''Normalize fixed_params to standard dict format.'''
    if fixed_params is None:
        # No fixed parameters requested.
        return {}

    if isinstance(fixed_params, dict):
        # Normalize each model entry to a set for fast membership checks.
        return {
            model: set(params) if isinstance(params, (list, tuple, set)) else {params}
            for model, params in fixed_params.items()
        }

    if isinstance(fixed_params, (list, tuple)):
        # Convert list of (model, param) pairs into dict-of-sets format.
        result = {}
        for item in fixed_params:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    f"Invalid fixed_params format: {item}. "
                    "Expected (model_name, param_name) tuples."
                )
            model, param = item
            if model not in result:
                result[model] = set()
            result[model].add(param)
        return result

    raise TypeError("fixed_params must be dict, list of tuples, or None")


def _merge_bounds_with_defaults(model_name, user_bounds, default_bounds):
    '''Merge user-provided bounds with default bounds for one model.'''
    merged = default_bounds.copy()

    if user_bounds is not None:
        for param_name, bound in user_bounds.items():
            if param_name not in merged:
                raise ValueError(
                    f"Model '{model_name}': parameter '{param_name}' "
                    "is not recognized for this model type"
                )
            merged[param_name] = bound

    return merged


def _get_oscillator_base_models():
    return {
        model_type: globals()[func_name]
        for model_type, func_name in _OSCILLATOR_BASE_MODELS.items()
    }


def _normalize_oscillator_input(oscillator):
    if isinstance(oscillator, dict):
        return {
            model_name: dict(model_dict)
            for model_name, model_dict in oscillator.items()
        }

    if isinstance(oscillator, (list, tuple)):
        normalized = {}
        counts = {}
        for idx, model_dict in enumerate(oscillator):
            if not isinstance(model_dict, dict):
                raise TypeError(
                    f"oscillator[{idx}] must be a dict with a 'type' key"
                )
            if 'type' not in model_dict:
                raise ValueError(
                    f"oscillator[{idx}] is missing required 'type' key"
                )
            model_type = str(model_dict['type']).lower()
            counts[model_type] = counts.get(model_type, 0) + 1
            normalized[f"{model_type}_{counts[model_type]}"] = dict(model_dict)
        return normalized

    raise TypeError("oscillator must be a dict or a list of model dicts")


def _build_oscillator_metadata(oscillator, *, bounds=None, fixed_params=None):
    '''
    Normalize oscillator specs and map free parameters into one optimizer vector.

    Each model entry stores the original parameter order, fixed values, bounds,
    and vector locations needed by fitting and interactive sliders.
    '''
    oscillator_dict = _normalize_oscillator_input(oscillator)
    if bounds is not None and not isinstance(bounds, dict):
        raise TypeError("bounds must be a dictionary or None")

    base_models = _get_oscillator_base_models()
    fixed_params_dict = _normalize_fixed_params(fixed_params)
    model_entries = []
    p0 = []
    lb = []
    ub = []

    for model_name, model_dict in oscillator_dict.items():
        if 'type' not in model_dict:
            raise ValueError(
                f"Model '{model_name}' is missing required 'type' key. "
                "Each model must define one of: drude, lorentz, tauc-lorentz, gaussian."
            )

        model_type = str(model_dict['type']).lower()
        if model_type not in base_models:
            raise ValueError(
                f"Model '{model_name}': type '{model_type}' is not recognized. "
                f"Valid types are: {list(base_models.keys())}"
            )

        func = base_models[model_type]
        sig = _inspect.signature(func)
        required_params = list(sig.parameters.keys())[1:]
        model_params = {k: v for k, v in model_dict.items() if k != 'type'}

        missing_params = set(required_params) - set(model_params.keys())
        extra_params = set(model_params.keys()) - set(required_params)
        if missing_params or extra_params:
            raise ValueError(
                f"Model '{model_name}' (type: {model_type}) requires parameters: "
                f"{required_params}, but got: {list(model_params.keys())}"
            )

        user_bounds_for_model = bounds.get(model_name) if bounds else None
        model_bounds = _merge_bounds_with_defaults(
            model_name,
            user_bounds_for_model,
            _OSCILLATOR_DEFAULT_BOUNDS[model_type]
        )

        param_values = _np.array(
            [float(model_params[param_name]) for param_name in required_params],
            dtype=float
        )
        model_fixed = fixed_params_dict.get(model_name, set())
        free_param_locs = []
        free_vector_locs = []
        for param_idx, param_name in enumerate(required_params):
            if param_name in model_fixed:
                continue
            vector_idx = len(p0)
            param_value = float(model_params[param_name])
            lower, upper = model_bounds[param_name]
            p0.append(param_value)
            lb.append(float(lower))
            ub.append(float(upper))
            free_param_locs.append(param_idx)
            free_vector_locs.append(vector_idx)

        model_entries.append({
            'name': model_name,
            'type': model_dict['type'],
            'func': func,
            'required_params': tuple(required_params),
            'base_values': param_values,
            'free_param_locs': _np.asarray(free_param_locs, dtype=int),
            'free_vector_locs': _np.asarray(free_vector_locs, dtype=int),
            'bounds': model_bounds,
        })

    return {
        'oscillator_dict': oscillator_dict,
        'fixed_params_dict': fixed_params_dict,
        'model_entries': model_entries,
        'p0': _np.asarray(p0, dtype=float),
        'lower_bounds': _np.asarray(lb, dtype=float),
        'upper_bounds': _np.asarray(ub, dtype=float),
    }


def _construct_oscillator_dict_from_entries(model_entries, p):
    out = {}
    for entry in model_entries:
        values = entry['base_values'].copy()
        if entry['free_vector_locs'].size > 0:
            values[entry['free_param_locs']] = p[entry['free_vector_locs']]
        out[entry['name']] = {'type': entry['type']}
        for param_name, value in zip(entry['required_params'], values):
            out[entry['name']][param_name] = float(value)
    return out


def _evaluate_nk_from_entries(lam, model_entries, p):
    eps = complex(0, 0)
    for entry in model_entries:
        values = entry['base_values']
        if entry['free_vector_locs'].size > 0:
            values = values.copy()
            values[entry['free_param_locs']] = p[entry['free_vector_locs']]
        eps += entry['func'](lam, *values) ** 2
    return _np.sqrt(eps)


def _as_interactive_model_block(arr, name, lam):
    '''Validate a custom interactive model block against the plotted wavelength grid.'''
    if hasattr(arr, 'index'):
        try:
            index_values = _np.asarray(arr.index, dtype=float)
        except Exception:
            index_values = None
        if index_values is not None and len(index_values) == len(lam):
            if not _np.allclose(index_values, lam, rtol=1e-12, atol=1e-12):
                raise ValueError(
                    f"{name} has a pandas index that does not match wavelength. "
                    "interactive_oscillator_guess plots model outputs against the "
                    "wavelength argument, so y_eval must return numpy arrays or "
                    "pandas Series indexed by wavelength."
                )

    out = _as_1d_real(arr, name)
    if len(out) != len(lam):
        raise ValueError(
            f"{name} has length {len(out)} but wavelength has length {len(lam)}"
        )
    return out


def _evaluate_model_blocks(context, p, *, extra_kwargs=None, interactive=False,
                           require_target_lengths=True, y_eval_min_extra_args=0):
    '''
    Evaluate oscillator output and return blocks aligned with prepared targets.

    Direct mode compares nk.real/nk.imag to parsed n/k columns. Custom mode calls
    y_eval once and validates that its one-or-many outputs match y_data blocks.
    '''
    lam = context['lam']
    data_blocks = context['data_blocks']
    nk = _evaluate_nk_from_entries(lam, context['model_entries'], p)

    if context['legacy_mode']:
        model_blocks = [
            _as_1d_real(nk.real if kind == 'n' else nk.imag, f'model_{kind}')
            for kind in context['direct_kinds']
        ]
    else:
        try:
            model_out = context['y_eval'](
                lam,
                nk,
                *context['eval_args'],
                **({} if extra_kwargs is None else extra_kwargs),
            )
        except TypeError as exc:
            if y_eval_min_extra_args > 0:
                raise RuntimeError(
                    "y_eval failed due to argument mismatch; verify args matches "
                    "y_eval(lam, nk, *args)"
                ) from exc
            raise RuntimeError(f"y_eval failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"y_eval failed: {exc}") from exc

        model_out_blocks = list(model_out) if isinstance(model_out, (list, tuple)) else [model_out]
        if len(model_out_blocks) != len(data_blocks):
            raise ValueError(
                f"y_eval returned {len(model_out_blocks)} outputs but y_data has "
                f"{len(data_blocks)} target blocks"
            )
        if interactive:
            model_blocks = [
                _as_interactive_model_block(model_out_blocks[i], f'model_output[{i}]', lam)
                for i in range(len(model_out_blocks))
            ]
        else:
            model_blocks = [
                _as_1d_real(model_out_blocks[i], f'model_output[{i}]')
                for i in range(len(model_out_blocks))
            ]

    if len(model_blocks) != len(data_blocks):
        raise ValueError("Model/data block count mismatch")

    same_lengths = all(len(model_i) == len(data_i) for model_i, data_i in zip(model_blocks, data_blocks))
    if require_target_lengths and not same_lengths:
        for i, (model_i, data_i) in enumerate(zip(model_blocks, data_blocks)):
            if len(model_i) != len(data_i):
                raise ValueError(
                    f"Target length mismatch at index {i}: "
                    f"model has {len(model_i)} points, data has {len(data_i)} points"
                )

    mse = (
        float(_np.mean(_np.concatenate([
            (model_i - data_i) ** 2 for model_i, data_i in zip(model_blocks, data_blocks)
        ])))
        if same_lengths else _np.nan
    )
    return {
        'nk': nk,
        'data_blocks': data_blocks,
        'model_blocks': model_blocks,
        'labels': context['target_labels'],
        'mse': mse,
    }


def multi_oscillator(wavelength, oscilator_dict):
    '''
    Computes refractive index using a combination of oscillator models.

    Parameters
    ----------
    wavelength : ndarray or float
        Wavelength range (um).
    oscilator_dict : dict
        Dictionary with named models containing a required 'type' key.
        Example:
        {
            'model1': {'type': 'drude', 'epsinf': 1.5, 'wp': 10.0, 'gamma': 0.1},
            'model2': {'type': 'lorentz', 'epsinf': 2.0, 'wp': 8.0, 'wn': 5.0, 'gamma': 0.05}
        }

    Returns
    -------
    ndarray (complex)
        Complex refractive index.
    '''
    base_models = {
        'tauc-lorentz': tauc_lorentz,
        'gaussian': gaussian,
        'lorentz': lorentz,
        'drude': drude
    }

    # Accumulate complex dielectric contributions from each named model.
    eps = complex(0, 0)
    for model_name, model_dict in oscilator_dict.items():
        if 'type' not in model_dict:
            raise ValueError(
                f"Model '{model_name}' is missing required 'type' key. "
                "Each model must define one of: drude, lorentz, tauc-lorentz, gaussian."
            )

        model_type = model_dict['type'].lower()
        if model_type not in base_models:
            raise ValueError(
                f"Model '{model_name}': type '{model_type}' is not recognized. "
                f"Valid types are: {list(base_models.keys())}"
            )

        params = {k: v for k, v in model_dict.items() if k != 'type'}
        sig = _inspect.signature(base_models[model_type])
        required_params = list(sig.parameters.keys())[1:]

        if set(params.keys()) != set(required_params):
            raise ValueError(
                f"Model '{model_name}' (type: {model_type}) requires parameters: "
                f"{required_params}, but got: {list(params.keys())}"
            )

        eps += base_models[model_type](wavelength, **params) ** 2

    # Return refractive index from total dielectric response.
    return _np.sqrt(eps)

def _as_1d_real(arr, name):
    # Coerce targets/model outputs to 1D real vectors used by least_squares.
    arr_1d = _as_1d_array(arr, name=name)
    arr_1d = _np.real_if_close(arr_1d, tol=1000)
    if _np.iscomplexobj(arr_1d):
        raise ValueError(f"{name} must be real-valued")
    return _np.asarray(arr_1d, float)

def _normalize_fit_extra_params(specs):
    if specs is None:
        return {}, []
    if not isinstance(specs, dict):
        raise TypeError("fit_extra_params must be a dictionary or None")

    normalized = {}
    order = []
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise TypeError(f"fit_extra_params['{name}'] must be a dict")
        if 'init' not in spec or 'bounds' not in spec:
            raise ValueError(
                f"fit_extra_params['{name}'] must contain 'init' and 'bounds'"
            )

        init_arr = _np.asarray(spec['init'], dtype=float)
        shape = spec.get('shape', None)
        if shape is None:
            shape = init_arr.shape
        else:
            shape = tuple(shape)

        if shape == ():
            init_arr = _np.asarray(float(init_arr.reshape(-1)[0]), dtype=float)
        else:
            init_arr = _np.array(_np.broadcast_to(init_arr, shape), dtype=float, copy=True)

        bounds = spec['bounds']
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"fit_extra_params['{name}']['bounds'] must be (lower, upper)"
            )

        lb_raw = _np.asarray(bounds[0], dtype=float)
        ub_raw = _np.asarray(bounds[1], dtype=float)

        if shape == ():
            lb_arr = _np.asarray(float(lb_raw.reshape(-1)[0]), dtype=float)
            ub_arr = _np.asarray(float(ub_raw.reshape(-1)[0]), dtype=float)
        else:
            lb_arr = _np.array(_np.broadcast_to(lb_raw, shape), dtype=float, copy=True)
            ub_arr = _np.array(_np.broadcast_to(ub_raw, shape), dtype=float, copy=True)

        init_flat = init_arr.reshape(-1).astype(float)
        lb_flat = lb_arr.reshape(-1).astype(float)
        ub_flat = ub_arr.reshape(-1).astype(float)

        if _np.any(lb_flat > ub_flat):
            raise ValueError(f"fit_extra_params['{name}']: lower bounds exceed upper bounds")
        if _np.any((init_flat < lb_flat) | (init_flat > ub_flat)):
            raise ValueError(
                f"fit_extra_params['{name}']: init must lie within bounds"
            )

        weight = float(spec.get('weight', 0.0))
        if weight < 0:
            raise ValueError(
                f"fit_extra_params['{name}']['weight'] must be >= 0"
            )

        normalized[name] = {
            'shape': shape,
            'init_flat': init_flat,
            'lb_flat': lb_flat,
            'ub_flat': ub_flat,
            'weight': weight,
        }
        order.append(name)
    return normalized, order

def _build_weight_block(weight_item, target_len, idx):
    # Expand scalar weights or validate per-point weight arrays.
    if _np.isscalar(weight_item):
        return _np.ones(target_len, dtype=float) * float(weight_item)
    w_block = _np.asarray(weight_item, float).reshape(-1)
    if len(w_block) != target_len:
        raise ValueError(
            f"weights[{idx}] has length {len(w_block)} but target length is {target_len}"
        )
    return w_block


def _parse_direct_nk_columns(y_data):
    '''
    Parse direct-mode y_data columns into paired n/k target descriptors.

    Columns may be plain n/k or sample-tagged forms such as n_sample1 and
    k (sample1). The output order is n, k for each discovered sample.
    '''
    if not isinstance(y_data, _pd.DataFrame):
        raise TypeError("y_data must be a pandas DataFrame")

    samples = []
    by_sample = {}
    for col in y_data.columns:
        col_label = str(col)
        stripped = col_label.strip()
        if not stripped:
            raise ValueError("Direct-mode y_data columns must start with n or k")

        kind = stripped[0].lower()
        if kind not in ('n', 'k'):
            raise ValueError(
                "Direct-mode y_data columns must start with n or k "
                f"(got {col_label!r})"
            )

        if len(stripped) > 1 and stripped[1] not in (' ', '_', '-', '('):
            raise ValueError(
                "Direct-mode y_data columns must use n/k as a lead name followed "
                f"by a separator (got {col_label!r})"
            )

        sample = stripped[1:].strip()
        if sample.startswith('(') and sample.endswith(')') and len(sample) >= 2:
            sample = sample[1:-1]
        sample = _re.sub(r'^[\s_\-\(]+|[\s_\-\)]+$', '', sample).strip()

        if sample not in by_sample:
            by_sample[sample] = {'sample': sample}
            samples.append(by_sample[sample])

        sample_info = by_sample[sample]
        if kind in sample_info:
            sample_name = sample if sample else "legacy n/k"
            raise ValueError(
                f"Duplicate direct-mode {kind} column for sample {sample_name!r}"
            )
        sample_info[kind] = col

    if not samples:
        raise ValueError("y_data must have at least one n/k sample pair")

    missing = []
    for sample_info in samples:
        for kind in ('n', 'k'):
            if kind not in sample_info:
                sample = sample_info['sample'] if sample_info['sample'] else "legacy n/k"
                missing.append(f"{kind} for {sample!r}")
    if missing:
        raise ValueError(
            "Direct-mode y_data requires each sample to have both n and k columns; "
            f"missing {missing}"
        )

    targets = []
    for sample_info in samples:
        targets.append({
            'sample': sample_info['sample'],
            'kind': 'n',
            'column': sample_info['n'],
            'label': str(sample_info['n']),
        })
        targets.append({
            'sample': sample_info['sample'],
            'kind': 'k',
            'column': sample_info['k'],
            'label': str(sample_info['k']),
        })

    return samples, targets


def _validate_train_data(wavelength, y_data, target_columns, wvl_units):
    '''
    Validate wavelength and tabular targets shared by fitting and guessing.

    Wavelength is converted to microns, while y_data index values are assumed
    to already be in microns. The requested wavelength grid must stay inside
    the measured index range so neither workflow extrapolates target data.

    Parameters
    ----------
        wavelength : numpy.ndarray
            Wavelength range for fitting
        y_data : pandas.DataFrame
            Training data.
        target_columns : list of str
            Columns in y_data to use for training.
        wvl_units : str
            Units of the wavelength values.
    Returns
    -------
        lam : numpy.ndarray
            Wavelength values in micrometers.
        y_data : pandas.DataFrame
            Training data with appropriate index.
        index_values : numpy.ndarray
            Index values of the training data.
    '''
    if type(wavelength) is not _np.ndarray:
        raise TypeError("wavelength must be a 1D numpy.ndarray")
    if wavelength.ndim != 1:
        raise ValueError("wavelength must be a 1D numpy.ndarray")
    if wavelength.size == 0:
        raise ValueError("wavelength must not be empty")

    try:
        lam = _convert_units(wavelength, wvl_units, to='um')
        lam = _np.asarray(lam, float)
    except Exception as exc:
        raise ValueError("wavelength must contain numeric values") from exc

    if lam.ndim != 1:
        raise ValueError("wavelength must be a 1D numpy.ndarray")
    if not _np.all(_np.isfinite(lam)):
        raise ValueError("wavelength must contain only finite values")

    if not isinstance(y_data, _pd.DataFrame):
        raise TypeError("y_data must be a pandas DataFrame")
    if y_data.index.name != 'wavelength':
        raise ValueError("y_data index name must be 'wavelength'")
    if y_data.empty:
        raise ValueError("y_data must not be empty")
    if not _pd.api.types.is_numeric_dtype(y_data.index):
        raise TypeError("y_data index must be numeric")
    if not y_data.index.is_unique:
        raise ValueError("y_data index must be unique")

    missing_columns = [col for col in target_columns if col not in y_data.columns]
    if missing_columns:
        raise ValueError(f"y_data is missing required column(s): {missing_columns}")

    index_values = _np.asarray(y_data.index, float)
    if index_values.size == 0:
        raise ValueError("y_data index must not be empty")
    if not _np.all(_np.isfinite(index_values)):
        raise ValueError("y_data index must contain only finite values")

    if lam.min() < index_values.min() or lam.max() > index_values.max():
        raise ValueError(
            "wavelength range must be within y_data index range; extrapolation is not allowed"
        )

    return lam, y_data, index_values


def _prepare_fit_wavelength_and_targets(wavelength, y_data, target_columns, wvl_units):
    '''Return fit wavelength and target blocks, interpolating y_data when needed.'''
    lam, y_data, index_values = _validate_train_data(
        wavelength,
        y_data,
        target_columns,
        wvl_units,
    )

    exact_grid_match = len(lam) == len(index_values) and _np.array_equal(lam, index_values)
    if not exact_grid_match and not y_data.index.is_monotonic_increasing:
        y_data = y_data.sort_index()
        index_values = _np.asarray(y_data.index, float)

    y_data_blocks = []
    for col in target_columns:
        source = _as_1d_real(y_data[col].to_numpy(), f"y_data['{col}']")
        if exact_grid_match:
            y_data_blocks.append(source)
        else:
            y_data_blocks.append(_np.interp(lam, index_values, source))

    return lam, y_data_blocks


def _prepare_interactive_wavelength_and_targets(wavelength, y_data, target_columns, wvl_units):
    '''Return model wavelength plus raw measured-grid targets for interactive plots.'''
    lam, y_data, index_values = _validate_train_data(
        wavelength, 
        y_data, 
        target_columns, 
        wvl_units
        )

    data_blocks = [
        _as_1d_real(y_data[col].to_numpy(), f"y_data['{col}']")
        for col in target_columns
    ]
    
    return lam, index_values, data_blocks


def _prepare_oscillator_fit_context(wavelength, y_data, oscillator_dict, y_eval, args,
                                    wvl_units, bounds, fixed_params, *, mode):
    '''
    Build the shared oscillator/data context for fitting and interactive guessing.

    The context contains parsed target metadata, prepared data blocks, normalized
    oscillator metadata, and the evaluator arguments needed by both workflows.
    '''
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    if mode not in ('fit', 'interactive'):
        raise ValueError("mode must be 'fit' or 'interactive'")

    legacy_mode = y_eval is None
    if legacy_mode:
        if len(args) > 0:
            raise ValueError("args can only be used when y_eval is provided")
        direct_samples, direct_targets = _parse_direct_nk_columns(y_data)
        target_columns = [target['column'] for target in direct_targets]
        target_labels = [target['label'] for target in direct_targets]
        direct_kinds = [target['kind'] for target in direct_targets]
    else:
        if not callable(y_eval):
            raise TypeError("y_eval must be callable")
        if not isinstance(y_data, _pd.DataFrame):
            raise TypeError("y_data must be a pandas DataFrame")
        direct_samples = None
        direct_targets = None
        direct_kinds = None
        target_columns = list(y_data.columns)
        if len(target_columns) == 0:
            raise ValueError("y_data must have at least one column when y_eval is provided")
        target_labels = [str(col) for col in target_columns]

    if mode == 'fit':
        lam, data_blocks = _prepare_fit_wavelength_and_targets(
            wavelength,
            y_data,
            target_columns,
            wvl_units,
        )
        data_x = lam
    else:
        lam, data_x, data_blocks = _prepare_interactive_wavelength_and_targets(
            wavelength,
            y_data,
            target_columns,
            wvl_units,
        )

    metadata = _build_oscillator_metadata(
        oscillator_dict,
        bounds=bounds,
        fixed_params=fixed_params,
    )

    return {
        'legacy_mode': legacy_mode,
        'y_eval': y_eval,
        'eval_args': args,
        'lam': lam,
        'data_x': data_x,
        'data_blocks': data_blocks,
        'target_columns': target_columns,
        'target_labels': target_labels,
        'direct_samples': direct_samples,
        'direct_targets': direct_targets,
        'direct_kinds': direct_kinds,
        'metadata': metadata,
        'model_entries': metadata['model_entries'],
        'p0': metadata['p0'],
        'lower_bounds': metadata['lower_bounds'],
        'upper_bounds': metadata['upper_bounds'],
    }


class _InteractiveAxesProxy:
    _REFRESHING_PREFIXES = ('set_', 'invert_')
    _REFRESHING_METHODS = {
        'autoscale',
        'autoscale_view',
        'clear',
        'cla',
        'grid',
        'legend',
        'margins',
        'relim',
    }

    def __init__(self, axis, refresh):
        self._axis = axis
        self._refresh = refresh

    def __getattr__(self, name):
        attr = getattr(self._axis, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            out = attr(*args, **kwargs)
            if name.startswith(self._REFRESHING_PREFIXES) or name in self._REFRESHING_METHODS:
                self._refresh()
            return out

        return _wrapped

    def __eq__(self, other):
        return self._axis is other or self._axis == other


class _InteractiveOscillatorController:
    def __init__(self, widget, output, sliders, model_entries, initial_p, fig, ax, pyplot, display_func=None):
        self.widget = widget
        self.output = output
        self.fig = fig
        self.mpl_ax = tuple(ax)
        self.ax = tuple(_InteractiveAxesProxy(axis, self.refresh) for axis in self.mpl_ax)
        self._pyplot = pyplot
        self._display_func = display_func
        self._sliders = sliders
        self._model_entries = model_entries
        self._initial_p = _np.asarray(initial_p, dtype=float)

    def _current_param_vector(self):
        p = self._initial_p.copy()
        for slider_info in self._sliders:
            p[slider_info['vector_idx']] = float(slider_info['slider'].value)
        return p

    @property
    def model(self):
        return self.get_model()

    def get_model(self):
        return _construct_oscillator_dict_from_entries(
            self._model_entries,
            self._current_param_vector()
        )

    def refresh(self):
        canvas = getattr(self.fig, 'canvas', None)
        if canvas is not None and hasattr(canvas, 'draw_idle'):
            canvas.draw_idle()
        if self.output is None:
            if self._display_func is not None:
                self._display_func(self.fig)
            else:
                self._pyplot.show()
            return
        if hasattr(self.output, 'clear_output'):
            self.output.clear_output(wait=True)
        with self.output:
            if self._display_func is not None:
                self._display_func(self.fig)
            else:
                self._pyplot.show()

    def close(self):
        for slider_info in self._sliders:
            slider = slider_info['slider']
            observer = slider_info.get('observer')
            if observer is not None and hasattr(slider, 'unobserve'):
                slider.unobserve(observer, names='value')
            slider.close()
        if self.output is not None and hasattr(self.output, 'close'):
            self.output.close()
        if self.fig is not None:
            self._pyplot.close(self.fig)


def interactive_oscillator_guess(wavelength, y_data, oscillator_dict, y_eval=None, args=(),
                                 wvl_units='um', bounds=None, fixed_params=None,
                                 figure_kwargs=None):
    '''
    Build an interactive notebook UI to manually tune oscillator parameters.

    Parameters
    ----------
    wavelength : numpy.ndarray
        1D wavelength array. Values are converted to microns using wvl_units.
    y_data : pandas.DataFrame
        Measured target data indexed by wavelength. The index name must be
        'wavelength', and index values are assumed to already be in microns.
        Measured data is plotted exactly on this index and is not interpolated.
        Wavelength controls only the oscillator/model evaluation grid and must
        lie within the y_data index range.
        - If y_eval is None: y_data must have one or more paired n/k column
          groups, such as 'n'/'k' or 'n (sample1)'/'k (sample1)'.
        - If y_eval is provided: all DataFrame columns are used in column order,
          with one column per output returned by y_eval.
    oscillator_dict : dict or list of dict
        Initial oscillator guess. A dict must match the multi_oscillator format.
        A list must contain typed model dicts and will be auto-named.
    y_eval : callable, optional
        Forward model called as y_eval(lam_um, nk, *args). If None, direct
        comparison to paired n/k columns is used.
    args : tuple, optional
        Extra positional arguments passed to y_eval.
    wvl_units : str, optional
        Units of wavelength (default 'um').
    bounds : dict, optional
        Partial bounds for slider ranges using the same shape as fit_to_oscillator bounds.
    fixed_params : dict, list, or None, optional
        Parameters to keep fixed; fixed parameters do not get sliders.
    figure_kwargs : dict, optional
        Extra keyword arguments forwarded to matplotlib.pyplot.subplots.

    Returns
    -------
    _InteractiveOscillatorController
        Controller exposing `.widget`, `.fig`, `.ax`, `.model`, `.get_model()`, and `.close()`.
    '''
    # Import dependencies here so that the main library does not require ipywidgets or matplotlib.
    try:
        from ipywidgets import HBox as _HBox, Label as _Label, VBox as _VBox, FloatSlider as _FloatSlider, Output as _Output
    except ImportError as exc:
        raise ImportError(
            "interactive_oscillator_guess requires ipywidgets. Install it with `pip install ipywidgets`."
        ) from exc

    try:
        import matplotlib.pyplot as _plt
    except ImportError as exc:
        raise ImportError(
            "interactive_oscillator_guess requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    try:
        from IPython import get_ipython as _get_ipython
        from IPython.display import display as _display
        if _get_ipython() is None:
            _display = None
    except ImportError:
        _display = None

    figure_kwargs = {} if figure_kwargs is None else dict(figure_kwargs)
    context = _prepare_oscillator_fit_context(
        wavelength,
        y_data,
        oscillator_dict,
        y_eval,
        args,
        wvl_units,
        bounds,
        fixed_params,
        mode='interactive',
    )
    lam = context['lam']
    data_x = context['data_x']
    data_blocks = context['data_blocks']
    direct_samples = context['direct_samples']
    direct_targets = context['direct_targets']
    model_entries = context['model_entries']
    p0 = context['p0']

    _evaluate_model_blocks(context, p0, interactive=True, require_target_lengths=False)

    slider_infos = []
    sliders_by_model = {}

    for entry in model_entries:
        model_sliders = []
        for param_idx, vector_idx in zip(entry['free_param_locs'], entry['free_vector_locs']):
            param_name = entry['required_params'][param_idx]
            lower, upper = entry['bounds'][param_name]
            step = max((float(upper) - float(lower)) / 200.0, 1e-6)
            slider_key = f"{entry['name']}:{param_name}"
            slider = _FloatSlider(
                value=float(entry['base_values'][param_idx]),
                min=float(lower),
                max=float(upper),
                step=step,
                description=f"{param_name}:",
                continuous_update=True,
            )
            slider_infos.append({
                'name': slider_key,
                'vector_idx': int(vector_idx),
                'slider': slider,
            })
            model_sliders.append(slider)
        sliders_by_model[entry['name']] = model_sliders

    def _current_p_from_sliders():
        p = p0.copy()
        for slider_info in slider_infos:
            p[slider_info['vector_idx']] = float(slider_info['slider'].value)
        return p

    fig, axes = _plt.subplots(1, 2, squeeze=False, **figure_kwargs)
    data_ax, nk_ax = axes.ravel()
    evaluated = _evaluate_model_blocks(context, p0, interactive=True, require_target_lengths=False)
    model_lines = []
    data_xlim = (float(data_x.min()), float(data_x.max()))
    if y_eval is None:
        k_ax, n_ax = data_ax, nk_ax
        target_index = {
            (target['sample'], target['kind']): i
            for i, target in enumerate(direct_targets)
        }
        for sample_info in direct_samples:
            sample = sample_info['sample']
            n_idx = target_index[(sample, 'n')]
            k_idx = target_index[(sample, 'k')]
            k_line_data = k_ax.plot(
                data_x,
                evaluated['data_blocks'][k_idx],
                '-',
                label=evaluated['labels'][k_idx],
            )[0]
            data_color = k_line_data.get_color()
            n_ax.plot(
                data_x,
                evaluated['data_blocks'][n_idx],
                '-',
                color=data_color,
                label=evaluated['labels'][n_idx],
            )
        k_line = k_ax.plot(lam, evaluated['nk'].imag, '--k', label='k (fit)')[0]
        n_line = n_ax.plot(lam, evaluated['nk'].real, '--k', label='n (fit)')[0]
        k_ax.set_ylabel('k')
        k_ax.set_xlabel('wavelength (um)')
        k_ax.set_xlim(*data_xlim)
        k_ax.set_yscale('log')
        k_ax.legend()
        n_ax.set_ylabel('n')
        n_ax.set_xlabel('wavelength (um)')
        n_ax.set_xlim(*data_xlim)
        n_ax.legend()
    else:
        for i in range(len(evaluated['data_blocks'])):
            label = evaluated['labels'][i]
            data_line = data_ax.plot(data_x, evaluated['data_blocks'][i], '-', label=f'{label} data')[0]
            model_lines.append(
                data_ax.plot(
                    lam,
                    evaluated['model_blocks'][i],
                    '--',
                    color=data_line.get_color(),
                    label=f'{label} model',
                )[0]
            )
        data_ax.set_ylabel('Spectral Value')
        data_ax.set_xlabel('wavelength (um)')
        data_ax.set_xlim(*data_xlim)
        data_ax.legend()
        n_line = nk_ax.plot(lam, evaluated['nk'].real, '-', label='n model')[0]
        k_line = nk_ax.plot(lam, evaluated['nk'].imag, '-', label='k model')[0]
        nk_ax.set_ylabel('n, k')
        nk_ax.set_xlabel('wavelength (um)')
        nk_ax.set_xlim(*data_xlim)
        nk_ax.legend()

    def _set_title(evaluated_blocks):
        if _np.isfinite(evaluated_blocks['mse']):
            title = (
                "Interactive oscillator guess | mse = "
                f"{evaluated_blocks['mse']:.4g}"
            )
        else:
            title = "Interactive oscillator guess"
        fig.suptitle(title)

    _set_title(evaluated)
    fig.tight_layout()

    def _update_plot(change=None):
        p = _current_p_from_sliders()
        evaluated = _evaluate_model_blocks(context, p, interactive=True, require_target_lengths=False)
        for line, y_block in zip(model_lines, evaluated['model_blocks']):
            line.set_ydata(y_block)
        n_line.set_ydata(evaluated['nk'].real)
        k_line.set_ydata(evaluated['nk'].imag)
        _set_title(evaluated)
        controller.refresh()

    output = _Output()
    model_rows = [
        _HBox([_Label(f"{model_name}:"), *model_sliders])
        for model_name, model_sliders in sliders_by_model.items()
    ]
    widget = _VBox([*model_rows, output])
    controller = _InteractiveOscillatorController(
        widget,
        output,
        slider_infos,
        model_entries,
        p0,
        fig,
        (data_ax, nk_ax),
        _plt,
        display_func=_display,
    )

    for slider_info in slider_infos:
        slider = slider_info['slider']
        if hasattr(slider, 'observe'):
            slider.observe(_update_plot, names='value')
            slider_info['observer'] = _update_plot

    controller.refresh()
    return controller

def fit_to_oscillator(wavelength, y_data,
                      oscillator_dict,
                      y_eval=None,
                      args=(),
                      bounds=None,
                      weights=None,
                      fixed_params=None,
                      fit_extra_params=None,
                      wvl_units='um',
                      least_squares_args=None,
                      verbose=0):
    '''
    Fit oscillator parameters to measured data.

    Parameters
    ----------
    wavelength : numpy.ndarray
        1D wavelength array. Values are converted to microns using wvl_units.
    y_data : pandas.DataFrame
        Measured target data indexed by wavelength. The index name must be
        'wavelength', and index values are assumed to already be in microns.
        If wavelength does not exactly match the index, target columns are
        interpolated to wavelength. Extrapolation is not allowed.
        - If y_eval is None: y_data must have one or more paired n/k column
          groups, such as 'n'/'k' or 'n (sample1)'/'k (sample1)'.
        - If y_eval is provided: all DataFrame columns are used in column order,
          with one column per output returned by y_eval.
    oscillator_dict : dict
        Dictionary with named models containing 'type' key and parameters.
        Format:
        {'model1': {'type': 'drude', 'epsinf': 1.5, 'wp': 10.0, 'gamma': 0.1}, ...}
    y_eval : callable, optional
        Custom evaluator function with signature f(lam, nk, *args)
        that returns one array or multiple arrays (tuple/list).
        Example: y_eval = fun_RT where fun_RT returns (R_model, T_model).
        If None, direct fitting to paired n/k columns is used.
    args : tuple, optional
        Extra arguments passed to y_eval following scipy convention.
        For example, y_eval(lam, nk, *args). Default is ().
    bounds : dict, optional
        Partial bounds for specific parameters. Missing bounds use defaults.
        Format: {'model1': {'epsinf': (0, 2)}, 'model2': {'wp': (5, 20)}}
    weights : array_like or scalar, optional
        Residual weights.
        - None: uniform weights.
        - scalar: same weight for all residuals.
        - Direct mode (y_eval is None): tuple/list of 2 entries applies to
          n and k across all samples; one entry per target column is also accepted.
        - Custom mode: tuple/list with one entry per target column in y_data.
          Each entry can be scalar or array matching target length.
    fixed_params : dict, list, or None, optional
        Parameters to keep fixed (not optimized).
        Supported formats:
        {'model1': ['gamma', 'wp']} or [('model1', 'gamma'), ('model2', 'wp')]
    fit_extra_params : dict, optional
        Extra y_eval parameters to fit in custom mode (y_eval must be provided).
        Format:
        {
            'param_name': {
                'init': value_or_array,            # required
                'bounds': (lower, upper),          # required
                'shape': tuple_or_list,            # optional, defaults to init shape
                'weight': float                    # optional, default 0.0
            }
        }
        The optional weight adds quadratic regularization terms to the residual:
        sqrt(weight) * (param - init).
    wvl_units : str, optional
        Units of wavelength (default: 'um').
    least_squares_args : dict, optional
        Additional keyword arguments forwarded to scipy.optimize.least_squares,
        such as {'method': 'trf', 'max_nfev': 500, 'loss': 'soft_l1'}.
        The arguments fun, x0, bounds, and verbose are controlled by
        fit_to_oscillator and cannot be passed here.
    verbose : int, optional
        Verbosity level for least_squares output (default: 0).

    Returns
    -------
    object
        Fitted oscillator result with model, lam_range, and lam_units attributes.
    OptimizeResult
        Output from scipy.optimize.least_squares.
        Additional attributes are attached:
        - fit_extra_params: structured fitted extra parameters passed to y_eval.
        - fit_extra_flat: flattened fitted extra parameters.
    '''

    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    eval_args = args

    if least_squares_args is None:
        least_squares_args = {}
    elif not isinstance(least_squares_args, dict):
        raise TypeError("least_squares_args must be a dict")
    else:
        least_squares_args = least_squares_args.copy()

    reserved_least_squares_args = {'fun', 'x0', 'bounds', 'verbose'}
    invalid_least_squares_args = reserved_least_squares_args.intersection(least_squares_args)
    if invalid_least_squares_args:
        raise ValueError(
            "least_squares_args cannot include arguments controlled by fit_to_oscillator: "
            f"{sorted(invalid_least_squares_args)}"
        )

    legacy_mode = y_eval is None
    y_eval_min_extra_args = 0

    fit_extra_specs, fit_extra_order = _normalize_fit_extra_params(fit_extra_params)

    if legacy_mode and len(fit_extra_order) > 0:
        raise ValueError("fit_extra_params can only be used when y_eval is provided")

    if not legacy_mode:
        # Fit-only validation: custom evaluators may receive fitted extra kwargs.
        if not callable(y_eval):
            raise TypeError("y_eval must be callable")

        # Check y_eval positional/keyword signature compatibility.
        sig = _inspect.signature(y_eval)
        params = list(sig.parameters.values())
        pos_params = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
        has_varkw = any(p.kind == p.VAR_KEYWORD for p in params)
        max_positional = None if has_varargs else len(pos_params)

        if max_positional is not None and max_positional < 2:
            raise TypeError("y_eval must accept at least two positional arguments: lam and nk")

        if max_positional is not None and len(eval_args) > (max_positional - 2):
            raise ValueError(
                f"y_eval accepts at most {max_positional - 2} extra positional arguments via args, "
                f"but got {len(eval_args)}"
            )

        fit_extra_names = set(fit_extra_order)
        accepted_kw_names = {
            p.name for p in params
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }

        if not has_varkw:
            invalid = fit_extra_names - accepted_kw_names
            if invalid:
                raise ValueError(
                    f"fit_extra_params has names not accepted by y_eval: {sorted(invalid)}"
                )

        # Positional parameters after lam, nk that are not supplied by args.
        remaining_pos = pos_params[2 + len(eval_args):] if not has_varargs else []
        for p in remaining_pos:
            if p.kind == p.POSITIONAL_ONLY and p.default is p.empty:
                raise ValueError(
                    f"y_eval requires positional-only argument '{p.name}' not provided by args"
                )
            if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is p.empty and p.name not in fit_extra_names:
                raise ValueError(
                    f"y_eval requires argument '{p.name}' not provided by args or fit_extra_params"
                )

        # Required keyword-only arguments must be provided by fit_extra_params.
        for p in params:
            if p.kind == p.KEYWORD_ONLY and p.default is p.empty and p.name not in fit_extra_names:
                raise ValueError(
                    f"y_eval requires keyword-only argument '{p.name}' not provided in fit_extra_params"
                )

        min_positional = len([p for p in pos_params if p.default is p.empty])
        y_eval_min_extra_args = max(0, min_positional - 2)

    context = _prepare_oscillator_fit_context(
        wavelength,
        y_data,
        oscillator_dict,
        y_eval,
        eval_args,
        wvl_units,
        bounds,
        fixed_params,
        mode='fit',
    )
    lam = context['lam']
    y_data_blocks = context['data_blocks']
    direct_kinds = context['direct_kinds']
    model_entries = context['model_entries']

    target_sizes = [len(arr) for arr in y_data_blocks]
    y = _np.concatenate(y_data_blocks)

    p0 = context['p0'].copy()
    lower_bounds = context['lower_bounds'].copy()
    upper_bounds = context['upper_bounds'].copy()
    fit_extra_entries = []

    # Add fit_extra_params entries to optimizer vector.
    for name in fit_extra_order:
        spec = fit_extra_specs[name]
        start = len(p0)
        n = spec['init_flat'].size
        fit_slice = slice(start, start + n)

        p0 = _np.concatenate([p0, spec['init_flat']])
        lower_bounds = _np.concatenate([lower_bounds, spec['lb_flat']])
        upper_bounds = _np.concatenate([upper_bounds, spec['ub_flat']])
        fit_extra_entries.append({
            'name': name,
            'shape': spec['shape'],
            'slice': fit_slice,
            'init_flat': spec['init_flat'],
            'weight': spec['weight'],
        })

    p0 = _np.asarray(p0, float)
    p_bounds = (lower_bounds, upper_bounds)
    n_osc_params = sum(entry['free_vector_locs'].size for entry in model_entries)
    n_fit_extra_params = sum(fit_extra_specs[name]['init_flat'].size for name in fit_extra_order)

    if weights is None:
        w = _np.ones_like(y, dtype=float)
    elif _np.isscalar(weights):
        w = _np.ones_like(y, dtype=float) * float(weights)
    elif isinstance(weights, _np.ndarray):
        w = _np.asarray(weights, float).reshape(-1)
        if len(w) != len(y):
            raise ValueError(f"weights length ({len(w)}) must match residual length ({len(y)})")
    elif isinstance(weights, (list, tuple)):
        if legacy_mode and len(weights) == 2:
            direct_weight_blocks = []
            for i, kind in enumerate(direct_kinds):
                weight_idx = 0 if kind == 'n' else 1
                direct_weight_blocks.append(
                    _build_weight_block(weights[weight_idx], target_sizes[i], i)
                )
            w = _np.concatenate(direct_weight_blocks)
        else:
            if len(weights) != len(target_sizes):
                raise ValueError(
                    "weights must have one entry per target block in y_data "
                    f"(expected {len(target_sizes)}, got {len(weights)})"
                )
            w = _np.concatenate([
                _build_weight_block(weights[i], target_sizes[i], i)
                for i in range(len(target_sizes))
            ])
    else:
        raise TypeError("weights must be None, scalar, ndarray, or list/tuple")

    def construct_fit_extra_dict(p):
        out = {}
        for entry in fit_extra_entries:
            vals = _np.asarray(p[entry['slice']], dtype=float)
            shape = entry['shape']
            if shape == ():
                out[entry['name']] = float(vals[0])
            else:
                out[entry['name']] = vals.reshape(shape)
        return out

    def construct_oscillator_dict(p):
        return _construct_oscillator_dict_from_entries(model_entries, p)

    def resid(p):
        # Shared evaluator returns blocks in the same order as y_data_blocks.
        evaluated = _evaluate_model_blocks(
            context,
            p,
            extra_kwargs=construct_fit_extra_dict(p),
            y_eval_min_extra_args=y_eval_min_extra_args,
        )
        model_blocks = evaluated['model_blocks']

        r = _np.concatenate([
            model_i - data_i for model_i, data_i in zip(model_blocks, y_data_blocks)
        ])
        weighted = r * w

        # Optional quadratic regularization around init for fit_extra params.
        reg_terms = []
        for entry in fit_extra_entries:
            if entry['weight'] > 0:
                vals = _np.asarray(p[entry['slice']], dtype=float)
                reg = _np.sqrt(entry['weight']) * (vals - entry['init_flat'])
                reg_terms.append(reg)

        if reg_terms:
            return _np.concatenate([weighted] + reg_terms)
        return weighted

    if p0.size == 0:
        # Shortcut when every parameter is fixed.
        class _EmptyResult:
            pass
        res = _EmptyResult()
        res.x = _np.array([], dtype=float)
        res.success = True
        res.cost = 0.5 * _np.sum(resid(_np.array([], dtype=float)) ** 2)
        res.fit_extra_params = {}
        res.fit_extra_flat = _np.array([], dtype=float)
    else:
        res = _least_squares(resid, p0, bounds=p_bounds, verbose=verbose, **least_squares_args)

        res.fit_extra_params = construct_fit_extra_dict(res.x)
        if n_fit_extra_params > 0:
            res.fit_extra_flat = _np.asarray(res.x[n_osc_params:n_osc_params + n_fit_extra_params], dtype=float)
        else:
            res.fit_extra_flat = _np.array([], dtype=float)

    if p0.size == 0 and len(fit_extra_order) > 0:
        # Populate metadata for fixed-only path if ever used with non-empty extras.
        res.fit_extra_params = {
            name: (float(fit_extra_specs[name]['init_flat'][0]) if fit_extra_specs[name]['shape'] == ()
                   else fit_extra_specs[name]['init_flat'].reshape(fit_extra_specs[name]['shape']))
            for name in fit_extra_order
        }
        res.fit_extra_flat = _np.concatenate([fit_extra_specs[name]['init_flat'] for name in fit_extra_order])

    # store results into a class with attributes for the fitted oscillator parameters and lam range
    class _OscillatorFit:
        pass

    oscillator_fit = _OscillatorFit()
    oscillator_fit.model = construct_oscillator_dict(res.x)
    oscillator_fit.lam_range = (float(lam.min()), float(lam.max()))
    oscillator_fit.lam_units = 'um'

    return oscillator_fit, res

def print_oscillator_params(oscillator_dict):
    '''
    Utility function to print oscillator parameters in a readable format.
    '''
    for model_name, model_dict in oscillator_dict.items():
        print(f"Model: {model_name} (type: {model_dict['type']})")
        for param_name, value in model_dict.items():
            if param_name != 'type':
                print(f"  {param_name}: {value:.3f} eV")
        print()

def emt_multilayer_sphere(D: _List[float],
                          Np: _List[_Union[float, _np.ndarray]],
                          *,
                          check_inputs=True):
    '''
    Effective refractive index of a multilayer sphere using Bruggeman EMT.
    
    Parameters
    ----------
    D_layers: _List[float]
        List of layer thicknesses (in um)

    Np: _List[_Union[float, _np.ndarray]]
        List of refractive indices for each layer
    
    check_inputs: bool, optional
        If True, validate and preprocess inputs (default is True)

    Returns
    -------        
    N_eff: _np.ndarray
        Effective refractive index of the multilayer sphere
    '''
    if check_inputs:
        _, _,  Np, D, _ = _check_mie_inputs(Np_shells=Np, D=D)

    D = _np.asarray(D)           # ensure D is np array

    # Single layer case
    if len(D) == 1:
        return Np.reshape(-1)

    # Multilayer case: compute volume fractions and apply Bruggeman EMT
    R_layers = D / 2.0  # Convert to radii

    # Start with the innermost layer as the "host"
    N_eff = Np[0].copy()

    # Iteratively add each outer layer using Bruggeman EMT
    for i in range(1, len(D)):
        # Volume of current layer shell
        if i == 1:
            # First shell: volume from center to R_layers[1]
            V_total = (4/3) * _np.pi * R_layers[i]**3
            V_inner = (4/3) * _np.pi * R_layers[i-1]**3
        else:
            # Subsequent shells: volume of current composite + new shell
            V_total = (4/3) * _np.pi * R_layers[i]**3
            V_inner = (4/3) * _np.pi * R_layers[i-1]**3
        
        V_shell = V_total - V_inner
        
        # Volume fractions
        fv_shell = V_shell / V_total
        fv_inner = V_inner / V_total
        
        # Apply Bruggeman EMT: 
        # N_eff (previous composite) is now the "host"
        # Np[i] (current layer) is the "inclusion"
        N_eff = emt_brugg(fv_shell, Np[i], N_eff)
    
    return N_eff

def emt_brugg(fv_1,nk_1,nk_2):
    '''
    Effective permitivity based on Bruggersman theory
    
        Parameters
    ----------
    fv_1: float   
        filling fraction of material inclusions

    nk_1: ndarray
        refractive index of inclusions
    
    nk_2: ndarray
        refractive index of host

    Returns
    -------
    nk_eff: ndarray
        complex refractive index of effective media
    '''
    
    # check simple cases first
    if fv_1 == 0:     # no inclusions
        return nk_2
    elif fv_1 == 1:   # no host
        return nk_1

    # prepare variables
    fv_2 = 1 - fv_1
    eps_1, eps_2 = nk_1**2, nk_2**2 # convert refractive index to dielectric constants
    
    # check if eps_1 or eps_2 are scalar and convert both to 1D ndarray
    eps_1 = _as_1d_array(eps_1, name = "eps_1")
    eps_2 = _as_1d_array(eps_2, name = "eps_2")

    # eps_1 is scalar, create a constant array of len(eps_2)
    if len(eps_1) == 1 and len(eps_2) > 1:
        eps_1 = eps_1*_np.ones_like(eps_2)
        
    # eps_2 is scalar, create a constant array of len(eps_1)
    elif len(eps_2) == 1 and len(eps_1) > 1:
        eps_2 = eps_2*_np.ones_like(eps_1)
    
    # both are ndarrays, assert they have same length
    else:
        assert len(eps_1) == len(eps_2), 'size of eps_1 and eps_2 must be equal'

    # compute effective dielectric constant ussing Bruggerman theory.
    eps_m = 1/4.*((3*fv_1 - 1)*eps_1 + (3*fv_2 - 1)*eps_2                           \
            - _np.sqrt(((3*fv_1 - 1)*eps_1 + (3*fv_2 - 1)*eps_2)**2 + 8*eps_1*eps_2))
    
    for i in range(len(eps_m)):
        if eps_m[i].imag < 0  or (eps_m[i].imag < 1E-10 and eps_m[i].real < 0):
            eps_m[i] =  eps_m[i] + \
                1/2*_np.sqrt(((3*fv_1 - 1)*eps_1[i] + (3*fv_2 - 1)*eps_2[i])**2 \
                + 8*eps_1[i]*eps_2[i]) 
    
    # if eps_1 and eps_2 were scalar, return a single scalar value
    if len(eps_m) == 1: return _np.sqrt(eps_m[0])
    else :              return _np.sqrt(eps_m)

def eps_real_kkr(wavelength, eps_imag, eps_inf = 0, int_range = (0, _np.inf), cshift=1e-12):
    '''
    Computes real part of dielectric constant from its imaginary components 
    using Krammers-Kronig relation

    Parameters
    ----------
    wavelength: ndarray or float
         wavelength spectrum (in microns)
    
    eps_imag: ndarray, float or callable 
              imaginary component of refractive index (if ndarray, it must be same size as wavelength)
    
    eps_inf: float (default 0)
             dielectric constant at infinity
    
    int_range: 2D tupple (default 0, inf) 
               integration range (only for eps_inf is callable)
    
    cshift: float
            Small value to avoid singularity at integration

    Returns
    -------
    eps_real: ndarray or float
              real part of dielectric constant
    '''
    wavelength = _as_1d_array(wavelength, name="wavelength")
    cshift = complex(0, cshift)
    w_i = _convert_units(wavelength, 'um', 'eV')

    if  isinstance(eps_imag, _Callable):
        a, b = int_range # set integration range
        def integration_element(w_r):
            factor = lambda w: w / (w**2 - w_r**2 + cshift)
            real_int = lambda w: (eps_imag(w) * factor(w)).real
            imag_int = lambda w: (eps_imag(w) * factor(w)).imag
            total = _quad(real_int, a,b)[0] + 1j*_quad(imag_int, a,b)[0]
            return eps_inf + (2/_np.pi)*total
        
    elif isinstance(eps_imag, _np.ndarray) or isinstance(eps_imag,float):
        eps_imag = _as_1d_array(eps_imag, name="eps_imag")
        assert wavelength.shape == eps_imag.shape, 'input arrays must be same length'
    
        def integration_element(w_r):
            factor = - w_i / (w_i**2 - w_r**2 + cshift) # integration domains are swaped, so a "-"" sign is added
            total = _np.trapz(eps_imag * factor, x=w_i)
            return eps_inf + (2/_np.pi)*total
    else:
        raise TypeError('Unknown type for eps_imag')
    
    eps_real = _np.real([integration_element(w_r) for w_r in w_i]).reshape(-1)
    
    return float(eps_real[0]) if len(wavelength) == 1 else eps_real 
'''
    --------------------------------------------------------------------
                            Target functions
    --------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
#                                   Inorganic
# refractive index of SiO2 (quartz)
# SiO2 = lambda wavelength: get_nkfile(wavelength, 'sio2_Palik_Lemarchand2013', get_from_local_path = True)[0]
SiO2 = lambda wavelength: get_ri_info(wavelength, 'main', 'SiO2', 'Franta-25C')[0]

# refractive index of Fused silica
Silica = lambda wavelength: get_ri_info(wavelength, 'main', 'SiO2', 'Franta')[0]

# refractive index of CaCO3
CaCO3 = lambda wavelength: get_nkfile(wavelength, 'CaCO3_Palik', get_from_local_path = True)[0]

# refractive index of BaSO4
BaSO4 = lambda wavelength: get_nkfile(wavelength, 'BaSO4_Tong2022', get_from_local_path = True)[0]

# refractive index of BaF2
BaF2 = lambda wavelength: get_ri_info(wavelength, 'main', 'BaF2', 'Querry')[0]

# refractive index of TiO2
TiO2 = lambda wavelength: get_ri_info(wavelength,'main','TiO2','Siefke')[0]

# refractive index of BiVO4 monoclinic (a axis)
BiVO4_mono_a = lambda wavelength: get_nkfile(wavelength, 'BiVO4_a-c_Zhao2011', get_from_local_path = True)[0]

# refractive index of BiVO4 monoclinic (b axis)
BiVO4_mono_b = lambda wavelength: get_nkfile(wavelength, 'BiVO4_b_Zhao2011', get_from_local_path = True)[0]

# refractive index of BiVO4 monoclinic (c axis)
BiVO4_mono_c = lambda wavelength: get_nkfile(wavelength, 'BiVO4_a-c_Zhao2011', get_from_local_path = True)[0]

# average refractive index of BiVO4 monoclinic
BiVO4 = lambda wavelength: (BiVO4_mono_a(wavelength) + BiVO4_mono_b(wavelength) + BiVO4_mono_c(wavelength))/3

# refractive index of Cu2O
Cu2O = lambda wavelength: get_nkfile(wavelength, 'Cu2O_Malerba2011', get_from_local_path = True)[0]

# refractive index of ZnO
ZnO = lambda wavelength: get_ri_info(wavelength,'main','ZnO','Querry')[0]

# refractive index of MgO
MgO = lambda wavelength: get_nkfile(wavelength,'MgO_Palik', get_from_local_path = True)[0]

# refractive index of MgF2
MgF2 = lambda wavelength: get_ri_info(wavelength,'main','MgF2','Franta')[0]

# refractive index of ZrO2
ZrO2 = lambda wavelength: get_ri_info(wavelength,'main','ZrO2','Synowicki')[0]

Al2O3 = lambda wavelength: get_ri_info(wavelength,'main','Al2O3','Querry-o')[0]

# refractive index of ZnS
ZnS = lambda wavelength: get_ri_info(wavelength,'main','ZnS','Querry')[0]

# refractive index of amorphous GeSbTe (GST)
GSTa = lambda wavelength: get_nkfile(wavelength, 'GSTa_Du2016', get_from_local_path = True)[0]

# refractive index of crystaline GeSbTe (GST)
GSTc = lambda wavelength: get_nkfile(wavelength, 'GSTc_Du2016', get_from_local_path = True)[0]

# refractive index of Monoclinic(cold) Vanadium Dioxide (VO2M)
# sputtered on SiO2 by default (film2)
VO2M = lambda wavelength, film = 2: get_nkfile(wavelength, 'VO2M_Wan2019(film%i)' % film, get_from_local_path = True)[0]

# refractive index of Rutile(hot) Vanadium Dioxide (VO2R)
# sputtered on SiO2 by default (film2)
VO2R = lambda wavelength, film = 2: get_nkfile(wavelength, 'VO2R_Wan2019(film%i)' % film, get_from_local_path = True)[0]

def VO2(wavelength, T, film=2 , Tphc = 73, WT = 3.1):
    '''
    Refractive index of temperatura dependent VO2.
    Reference: Wan, C. et al. Ann. Phys. 531, 1900188 (2019).

    Parameters
    ----------
    wavelength : ndarray
        Wavelength range (um).
    T : float
        Temperature of VO2 (°C).
    film : int, optional
        Film type according to reference (The default is 2):
         - film 1: Si+native oxide/VO2(70nm) (Sputtered). 
         - film 2: Si+native oxide/VO2(130nm) (Sputtered).
         - film 3: Saphire/VO2(120nm) (Sputtered). 
         - film 4: Si+native oxide/VO2(110nm) (Sol-gel). 
    Tphc : float, optional
        Transition temperature (°C). The default is 73.
    WT : float, optional
        Width of IMT phase change (ev). The default is 3.1.

    Returns
    -------
    Complex refractive index

    '''
    # set constants
    kB = 8.617333262E-5 # eV/K (Boltzmann constant)
    Tphc = Tphc + 273   # convert °C to K
    T = T + 273         # convert °C to K
    
    fv = 1/(1 + _np.exp(WT/kB*(1/T - 1/Tphc)))
    eps_c = VO2M(wavelength, film)**2
    eps_h = VO2R(wavelength, film)**2
    
    eps = (1 - fv)*eps_c + fv*eps_h
    
    return _np.sqrt(eps)

# refractive index of Silicon
Si   = lambda wavelength: get_ri_info(wavelength, 'main', 'Si', 'Franta-300K')[0]

#------------------------------------------------------------------------------
#                                   Metals
# refractive index of Gold
gold = lambda wavelength: get_nkfile(wavelength, 'au_Olmon2012_evap', get_from_local_path = True)[0]

# refractive index of Silver
silver = lambda wavelength: get_nkfile(wavelength, 'ag_Ciesielski2017', get_from_local_path = True)[0]

# refractive index of Copper
Cu   = lambda wavelength: get_nkfile(wavelength, 'cu_Babar2015', get_from_local_path = True)[0]

# refractive index of Aluminium
Al   = lambda wavelength: get_nkfile(wavelength, 'al_Rakic1995', get_from_local_path = True)[0]

# refractive index of Magnesium
Mg   = lambda wavelength: get_ri_info(wavelength, 'main', 'Mg', 'Hagemann')[0]

#------------------------------------------------------------------------------
#                                   Polymers
# refractive index of HDPE
HDPE  = lambda wavelength: get_nkfile(wavelength, 'HDPE_Palik', get_from_local_path = True)[0]

# refractive index of HDPE
PDMS  = lambda wavelength: get_nkfile(wavelength, 'PDMS_Zhang2020_Querry1987', get_from_local_path = True)[0]

# refractive index of PMMA
PMMA = lambda wavelength: get_ri_info(wavelength,'organic','(C5H8O2)n - poly(methyl methacrylate)','Zhang-Tomson')[0]

# refractive index of PVDF-HFP
PVDF  = lambda wavelength: get_nkfile(wavelength, 'PVDF-HFP_Mandal2018', get_from_local_path = True)[0]

# refractive index of Polystyrene
PS = lambda wavelength: get_ri_info(wavelength,'organic','(C8H8)n - polystyrene','Zhang')[0]

#------------------------------------------------------------------------------
#                                   Others
# refractive index of water
H2O  = lambda wavelength: get_nkfile(wavelength, 'h2o_Hale1973', get_from_local_path = True)[0]
