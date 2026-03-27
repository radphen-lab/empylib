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

__all__ = ('get_nkfile', 'get_ri_info', 
           'lorentz', 'drude', 'tauc_lorentz', 'gaussian', 'multi_oscillator', 'fit_to_oscillator',
           'emt_multilayer_sphere', 'emt_brugg', 'eps_real_kkr',
           'SiO2', 'Silica', 'CaCO3', 'BaSO4', 'BaF2', 'TiO2',
           'BiVO4_mono_a', 'BiVO4_mono_b', 'BiVO4_mono_c', 'BiVO4', 'Cu2O', 'ZnO',
            'MgO', 'Al2O3', 'ZnS', 'GSTa', 'GSTc', 'VO2M', 'VO2R', 'VO2',
            'Si', 'gold', 'silver', 'Cu', 'Al', 'Mg',
            'HDPE', 'PDMS', 'PMMA', 'PVDF', 'H2O')

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

def fit_to_oscillator(x, y_data,
                      oscillator_dict,
                      y_eval=None,
                      args=(),
                      bounds=None,
                      weights=None,
                      fixed_params=None,
                      fit_extra_params=None,
                      x_units='um', 
                      least_squares_method='trf',
                      verbose=0):
    '''
    Fit oscillator parameters to measured data.

    Parameters
    ----------
    x : array_like
        Independent variable array (e.g., wavelength in microns).
    y_data : list or tuple
        Measured target data.
        - If y_eval is None: y_data must be [n_data, k_data].
                - If y_eval is provided: y_data must have one entry per output
                    returned by y_eval, e.g. [R_measured, T_measured].
    oscillator_dict : dict
        Dictionary with named models containing 'type' key and parameters.
        Format:
        {'model1': {'type': 'drude', 'epsinf': 1.5, 'wp': 10.0, 'gamma': 0.1}, ...}
    y_eval : callable, optional
        Custom evaluator function with signature f(lam, nk, *args)
        that returns one array or multiple arrays (tuple/list).
        Example: y_eval = fun_RT where fun_RT returns (R_model, T_model).
        If None, legacy fitting to [n_data, k_data] is used.
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
        - Legacy mode (y_eval is None): tuple/list of 2 entries for n and k.
        - Custom mode: tuple/list with one entry per target in y_data.
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
    x_units : str, optional
        Units of x (default: 'um').
    least_squares_method : str, optional
        Method for scipy.optimize.least_squares (default: 'trf').
    verbose : int, optional
        Verbosity level for least_squares output (default: 0).

    Returns
    -------
    dict
        Oscillator dictionary with fitted and fixed parameters.
    OptimizeResult
        Output from scipy.optimize.least_squares.
        Additional attributes are attached:
        - fit_extra_params: structured fitted extra parameters passed to y_eval.
        - fit_extra_flat: flattened fitted extra parameters.
    '''

    lam = _convert_units(x, x_units, to='um')
    lam = _np.asarray(lam, float)

    if isinstance(args, tuple):
        eval_args = args
    else:
        raise TypeError("args must be a tuple")

    legacy_mode = y_eval is None
    y_eval_fn = None
    y_eval_min_extra_args = 0

    fit_extra_specs, fit_extra_order = _normalize_fit_extra_params(fit_extra_params)

    if legacy_mode and len(fit_extra_order) > 0:
        raise ValueError("fit_extra_params can only be used when y_eval is provided")

    if legacy_mode:
        # Legacy API path: y_data is interpreted as [n_data, k_data].
        if not isinstance(y_data, (list, tuple)) or len(y_data) != 2:
            raise ValueError("When y_eval is None, y_data must be [n_data, k_data]")
        n_data = _as_1d_real(y_data[0], 'n_data')
        k_data = _as_1d_real(y_data[1], 'k_data')
        y_data_blocks = [n_data, k_data]
        if len(eval_args) > 0:
            raise ValueError("args can only be used when y_eval is provided")
    else:
        # Custom API path: y_eval is one forward model returning all target blocks.
        if not callable(y_eval):
            raise TypeError("y_eval must be callable")
        if not isinstance(y_data, (list, tuple)):
            raise TypeError("y_data must be a list or tuple when y_eval is provided")

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

        y_data_blocks = [_as_1d_real(arr, f'y_data[{i}]') for i, arr in enumerate(y_data)]
        y_eval_fn = y_eval
        min_positional = len([p for p in pos_params if p.default is p.empty])
        y_eval_min_extra_args = max(0, min_positional - 2)

    target_sizes = [len(arr) for arr in y_data_blocks]
    y = _np.concatenate(y_data_blocks)

    default_bounds = {
        'drude': {'epsinf': (0, 10), 'wp': (1E-5, 100), 'gamma': (1E-5, 10)},
        'lorentz': {'epsinf': (0, 10), 'wp': (1E-5, 100), 'wn': (1E-2, 10), 'gamma': (1E-5, 10)},
        'tauc-lorentz': {'A': (0, 10), 'C': (0, 10), 'E0': (0, 10), 'Eg': (0, 10)},
        'gaussian': {'A': (0, 10), 'Br': (0, 10), 'E0': (0, 10)}
    }

    base_models = {
        'tauc-lorentz': tauc_lorentz,
        'gaussian': gaussian,
        'lorentz': lorentz,
        'drude': drude
    }

    if bounds is not None and not isinstance(bounds, dict):
        raise TypeError("bounds must be a dictionary or None")

    fixed_params_dict = _normalize_fixed_params(fixed_params)

    p0 = []
    lb = []
    ub = []
    fitted_param_index = {}
    fit_extra_index = {}

    for model_name, model_dict in oscillator_dict.items():
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

        sig = _inspect.signature(base_models[model_type])
        required_params = list(sig.parameters.keys())[1:]
        model_params = {k: v for k, v in model_dict.items() if k != 'type'}

        missing_params = set(required_params) - set(model_params.keys())
        if missing_params:
            raise ValueError(
                f"Model '{model_name}' (type: {model_type}) is missing parameters: {missing_params}"
            )

        user_bounds_for_model = bounds.get(model_name) if bounds else None
        model_bounds = _merge_bounds_with_defaults(
            model_name,
            user_bounds_for_model,
            default_bounds[model_type]
        )

        model_fixed = fixed_params_dict.get(model_name, set())
        for param_name in required_params:
            if param_name in model_fixed:
                # Keep fixed parameters at their input values.
                continue

            # Register only free parameters in optimizer vectors.
            fitted_param_index[(model_name, param_name)] = len(p0)
            p0.append(float(model_params[param_name]))
            lb.append(float(model_bounds[param_name][0]))
            ub.append(float(model_bounds[param_name][1]))

    # Add fit_extra_params entries to optimizer vector.
    for name in fit_extra_order:
        spec = fit_extra_specs[name]
        start = len(p0)
        n = spec['init_flat'].size
        fit_extra_index[name] = (start, n)

        p0.extend(spec['init_flat'].tolist())
        lb.extend(spec['lb_flat'].tolist())
        ub.extend(spec['ub_flat'].tolist())

    p0 = _np.asarray(p0, float)
    p_bounds = (_np.asarray(lb, float), _np.asarray(ub, float))
    n_osc_params = len(fitted_param_index)
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
            w = _np.concatenate([
                _build_weight_block(weights[0], target_sizes[0], 0),
                _build_weight_block(weights[1], target_sizes[1], 1)
            ])
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

    def construct_oscillator_dict(p):
        # Rebuild a full oscillator_dict from current optimization vector p.
        out = {}
        for model_name, model_dict in oscillator_dict.items():
            out[model_name] = {'type': model_dict['type']}
            model_type = model_dict['type'].lower()
            required_params = list(_inspect.signature(base_models[model_type]).parameters.keys())[1:]
            for param_name in required_params:
                key = (model_name, param_name)
                if key in fitted_param_index:
                    out[model_name][param_name] = float(p[fitted_param_index[key]])
                else:
                    out[model_name][param_name] = float(model_dict[param_name])
        return out

    def construct_fit_extra_dict(p):
        out = {}
        for name in fit_extra_order:
            start, n = fit_extra_index[name]
            shape = fit_extra_specs[name]['shape']
            vals = _np.asarray(p[start:start+n], dtype=float)
            if shape == ():
                out[name] = float(vals[0])
            else:
                out[name] = vals.reshape(shape)
        return out

    def resid(p):
        # Compute nk from current parameters before mapping to measured targets.
        osc = construct_oscillator_dict(p)
        nk = multi_oscillator(lam, osc)

        if legacy_mode:
            # Compare fitted nk directly against n/k measured blocks.
            model_blocks = [
                _as_1d_real(nk.real, 'model_n'),
                _as_1d_real(nk.imag, 'model_k')
            ]
        else:
            extra_kwargs = construct_fit_extra_dict(p)
            # Apply one custom forward function and split its outputs into blocks.
            try:
                model_out = y_eval_fn(lam, nk, *eval_args, **extra_kwargs)
            except TypeError as exc:
                if y_eval_min_extra_args > 0:
                    raise RuntimeError(
                        "y_eval failed due to argument mismatch; verify args matches "
                        "y_eval(lam, nk, *args)"
                    ) from exc
                raise RuntimeError(f"y_eval failed: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"y_eval failed: {exc}") from exc

            if isinstance(model_out, (list, tuple)):
                model_out_blocks = list(model_out)
            else:
                model_out_blocks = [model_out]

            if len(model_out_blocks) != len(y_data_blocks):
                raise ValueError(
                    f"y_eval returned {len(model_out_blocks)} outputs but y_data has "
                    f"{len(y_data_blocks)} target blocks"
                )

            model_blocks = [
                _as_1d_real(model_out_blocks[i], f'model_output[{i}]')
                for i in range(len(model_out_blocks))
            ]

        for i, (model_i, data_i) in enumerate(zip(model_blocks, y_data_blocks)):
            if len(model_i) != len(data_i):
                raise ValueError(
                    f"Target length mismatch at index {i}: "
                    f"model has {len(model_i)} points, data has {len(data_i)} points"
                )

        r = _np.concatenate([
            model_i - data_i for model_i, data_i in zip(model_blocks, y_data_blocks)
        ])
        weighted = r * w

        # Optional quadratic regularization around init for fit_extra params.
        reg_terms = []
        for name in fit_extra_order:
            spec = fit_extra_specs[name]
            if spec['weight'] > 0:
                start, n = fit_extra_index[name]
                vals = _np.asarray(p[start:start+n], dtype=float)
                reg = _np.sqrt(spec['weight']) * (vals - spec['init_flat'])
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
        res = _least_squares(resid, p0, bounds=p_bounds, method=least_squares_method, verbose=verbose)

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

# refractive index of Alumina (AL2O3)
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

#------------------------------------------------------------------------------
#                                   Others
# refractive index of water
H2O  = lambda wavelength: get_nkfile(wavelength, 'h2o_Hale1973', get_from_local_path = True)[0]
