# -*- coding: utf-8 -*-
"""
Library of tabulated refractive index

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import platform
from pyexpat import model
from matplotlib.pylab import gamma
import numpy as _np 
import pandas as _pd
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import least_squares
from typing import Callable # used to check callable variables
from pathlib import Path
# import refidx as ri
from .utils import _ndarray_check, convert_units as _convert_units, _check_mie_inputs, _warn_extrapolation
from typing import List as _List, Union as _Union
import yaml
import requests
from io import StringIO
import inspect as _inspect

__all__ = ('get_nkfile', 'get_ri_info', 
           'lorentz', 'drude', 'tauc_lorentz', 'gaussian', 'multi_oscillator', 'fit_to_oscillator',
           'emt_multilayer_sphere', 'emt_brugg', 'eps_real_kkr',
           'SiO2', 'Silica', 'CaCO3', 'BaSO4', 'BaF2', 'TiO2',
           'BiVO4_mono_a', 'BiVO4_mono_b', 'BiVO4_mono_c', 'BiVO4', 'Cu2O', 'ZnO',
            'MgO', 'Al2O3', 'ZnS', 'GSTa', 'GSTc', 'VO2M', 'VO2R', 'VO2',
            'Si', 'gold', 'silver', 'Cu', 'Al', 'Mg',
            'HDPE', 'PDMS', 'PMMA', 'PVDF', 'H2O')

def blend_model(lam, nk_df, nk_model, blend_low=None, blend_high=None):

    # get inside index based on zero values of nk_interp
    lam_min, lam_max = float(nk_df.index[0]), float(nk_df.index[-1])
    inside = (lam >= lam_min) & (lam <= lam_max)

    # interpolate nk data
    nk_interp = _np.interp(lam, nk_df.index, nk_df['n'] + 1j*nk_df['k'])
    nk_out = _np.empty_like(nk_interp, dtype=complex)
    nk_out[~inside] = nk_model[~inside]     # outside data range: model
    nk_out[inside] = nk_interp[inside]    # inside data range: interpolated data
    
    if blend_low is None:
        # set value based on order of magnitude of lam_min
        blend_low = 10**_np.round(_np.log10(lam_min))

    if blend_high is None:
        # set value based on order of magnitude of lam_max
        blend_high = 10**_np.round(_np.log10(lam_max))

    # blend lower end data to smooth transition
    if blend_low > 0:
        # Blend in [wl_min, wl_min+blend_window] and [wl_max-blend_window, wl_max]
        bw = float(blend_low)

        # lower edge
        low = (lam >= lam_min) & (lam <= lam_min + bw)
        if _np.any(low):
            t = (lam[low] - lam_min) / bw  # 0..1
            # smoothstep
            s = t*t*(3 - 2*t)
            nk_itp = nk_interp[low] 
            nk_out[low] = (1 - s)*nk_model[low] + s*nk_itp

    # blend higher end data to smooth transition
    if blend_high > 0:
        bw = float(blend_high)
        # upper edge
        high = (lam <= lam_max) & (lam >= lam_max - bw)
        if _np.any(high):
            t = (lam_max - lam[high]) / bw  # 0..1
            s = t*t*(3 - 2*t)
            nk_itp = nk_interp[high]
            nk_out[high] = (1 - s)*nk_model[high] + s*nk_itp

    return nk_out

def get_nkfile(lam, MaterialName=None, get_from_local_path = False, lam_units = 'um', *, 
               extrapolate = 'flat'):
    '''
    Reads a tabulated *.nk file and returns an interpolated
    1D numpy array with the complex refractive index
    
    Parameters
    ----------
    lam : ndarray
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
        caller_directory = Path(__file__).parent / 'nk_files'
    else :
        # if function is called from working directory (where the function is called)
        caller_directory = Path.cwd()
    
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

    # convert lam to um
    nk_df.index = _convert_units(nk_df.index, lam_units, to='um')

    return _process_nk_data(lam, nk_df, MaterialName, extrapolate)

def ri_info_data(shelf,book,page):
    """
    Reads a YAML file containing 'nk' tabulated optical data from a URL and returns:
    - lam: ndarray of wavelengths
    - nk: ndarray of complex refractive indices (n + ik)
    """
    url_root = 'https://refractiveindex.info/database/data/' 
    url = url_root  + shelf + '/'  + book  + '/nk/' + page + '.yml'

    # Download YAML content
    response = requests.get(url)
    response.raise_for_status()

    # Parse YAML content
    yaml_data = yaml.safe_load(response.text)

    # Extract tabulated data block
    nk_text = yaml_data['DATA'][0]['data']

    # Read into DataFrame using regex-based separator
    nk_df = _pd.read_csv(StringIO(nk_text), sep=r'\s+', names=['wavelength', 'n', 'k'])
    nk_df.index = nk_df['wavelength']           # set wavelength as index
    nk_df = nk_df.drop(columns=['wavelength'])  # remove 'wavelength' column

    return nk_df

def get_ri_info(lam,shelf,book,page, *, extrapolate = 'flat'):
    '''
    Extract refractive index from refractiveindex.info database. This code
    uses the refidx package from Bejamin Vial (https://gitlab.com/benvial/refidx)

    Parameters
    ----------
    lam : ndarray
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

    return _process_nk_data(lam, nk_df, MaterialName, extrapolate)

def _process_nk_data(lam, nk_df, MaterialName, extrapolate):
    '''
    Process nk dataframe and interpolate to desired wavelengths.
    
    Parameters
    ----------
    lam : ndarray
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
    
    # check if lam is not ndarray
    lam, lam_isfloat = _ndarray_check(lam) 

    # create complex refractive index using interpolation form nkfile
    nk_df_complex = nk_df['n'] + 1j*nk_df['k']
    if extrapolate is False:
        N = _np.interp(lam, nk_df.index, nk_df_complex, 
                       left = complex(0, 0), right = complex(0, 0))
    
    elif extrapolate == 'flat':
        N = _np.interp(lam, nk_df.index, nk_df_complex)
    
    elif isinstance(extrapolate, dict):
        N_model = multi_oscillator(lam, extrapolate)
        N = blend_model(lam, nk_df, N_model)
    else:
        raise ValueError("Extrapolation method not recognized. Use False, 'flat', or dict with oscillator parameters.") 

    # if N.real or N.imag < 0, make it = 0
    N[N.real<0] = 0                + 1j*N[N.real<0].imag # real part = 0 (keep imag part)
    
    # warning if extrapolated values
    lo, hi = float(nk_df.index[0]), float(nk_df.index[-1])
    _warn_extrapolation(lam, lo, hi, label=MaterialName, quantity="refractive index")
    
    # if lam was float (orginaly), convert N to a complex value
    return complex(N[0]) if lam_isfloat else N, nk_df

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

def _fix_nk_anomalous(lam, n, k):
    '''
    PENDING
    Analyze nk to fix anomalous behaviors. In the case of n, it just makes 
    n = 0 if n < 0. For k, analyze beer-lambert transmittance of a 1 um film. 
    Adjust k that fall T_bl > T_threshold to a very low value

    Parameters
    ------------
    lam: ndarray
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
    a_coef = 4*_np.pi*k/lam                    # Absorption coefficient of film (1/um)
    T_bl = _np.exp(-a_coef*d)                  # Get Beer-Lambert transmittance
    idx_list = _split_by_max(T_bl, T_threst)  # Find index that pass threshold

    k_new = k.copy()
    for idx in idx_list:

        # Adjust k values to a linear regression with very large slope
        slope = 20                            # slope of the curve
        x0, y0 = _np.log(lam[idx[0] ]), _np.log(k[idx[0]])
        b_dw = y0 + slope*x0                  # find y-intersept of downward curve

        x0, y0 = _np.log(lam[idx[-1] ]), _np.log(k[idx[-1]])
        b_up = y0 - slope*x0                  # find y-intersept of upward curve

        # find intersection between the two curves
        x_intersect = _np.exp((b_dw - b_up)/(2*slope)) 
        idx_cut = _np.where(lam < x_intersect)[0][-1]  # index of intersection

        # create new k values with linear curves
        k_new[idx[0]:idx_cut]  = _np.exp(b_dw - slope*_np.log(lam[idx[0]:idx_cut]))
        k_new[idx_cut:idx[-1]] = _np.exp(b_up + slope*_np.log(lam[idx_cut:idx[-1]]))

    return n_new + 1j*k_new

def gaussian(lam, A,Br,E0):
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

    lam : ndarray
        Wavelengths range (um)

    Returns
    -------
    eps : ndarray (complex)
        Complex dielectric constant
    '''
    #  Gauss model as function of E (in eV)
    f = 0.5/_np.sqrt(_np.log(2)) # scaling constant
    eps_G = lambda E: A*_np.exp(-(f*(E - E0)/Br)**2) \
                    - A*_np.exp(-(f*(E + E0)/Br)**2)

    # get real and imaginary part of dielectric constant
    E = _convert_units(lam,'um','eV')                               # lambda range in eV
    a, b = max(E0 - 5*Br,0), E0 + 5*Br                             # integration range
    eps_re = eps_real_kkr(lam,eps_G,int_range=(a, b), cshift=1e-4) # get real part from KK
    eps_im = eps_G(E)                                              # imaginary component

    eps = eps_re + 1j*eps_im
    return _np.sqrt(eps)

def tauc_lorentz(lam, A,C,E0,Eg):
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
        
    lam : ndarray
        Wavelengths range (um)

    Returns
    -------
    eps : ndarray (complex)
        Complex dielectric constant
    '''
    
    #  Tauc-Lorentz model as function of E (in eV)
    eps_TL = lambda E: 1/E*A*E0*C*(E - Eg)**2/ \
                 ((E**2 - E0**2)**2 + C**2*E**2)*(E > Eg)
    
    # get real and imaginary part of dielectric constant
    E = _convert_units(lam,'um','eV')                                # lambda range in eV
    a, b = Eg-20*C, Eg + 20*C                                            # set integration range
    eps_re = eps_real_kkr(lam,eps_TL,int_range=(a, b), cshift=1e-3) # get real part from KK
    eps_im = eps_TL(E)                                              # imaginary component
    
    #------------------------------------------------------------------------------
    # # Alternative form (didn't work)
    #------------------------------------------------------------------------------
    # source: Luis V. Rodríguez-de Marcos and Juan I. Larruquert, 
    #           "Analytic optical-constant model derived from Tauc-Lorentz and Urbach tail," 
    #           Opt. Express 24, 28561-28572 (2016)
    # a = 0.09
    # b = E + 1j*a
    # d = _np.sqrt(E0**2 - (C/2)**2) - 1j*C/2
    # F = lambda x,y,z: ((Eg + x)**2*_np.log(Eg + x) - (Eg - x)**2*_np.log(Eg - x))/ \
    #                     x*(x**2 - y**2)*(x**2 - z**2)

    # eps = 1 + A*E0*C/_np.pi*(F(b,d,d.conjugate())) + F(d,d.conjugate(),b) + F(d.conjugate(),b,d)
    #------------------------------------------------------------------------------
    eps = eps_re + 1j*eps_im
    return _np.sqrt(eps)

def lorentz(lam, epsinf,wp,wn,gamma):
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
    lam : linear _np.array
        wavelength spectrum in um

    Returns
    -------
    complex refractive index

    '''
    from .utils import convert_units
    w = convert_units(lam,'um','eV')  # conver from um to eV 
    
    return _np.sqrt(epsinf + wp**2/(wn**2 - w**2 - 1j*gamma*w))

def drude(lam, epsinf,wp,gamma):
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
    lam : linear _np.array
        wavelength spectrum in um

    Returns
    -------
    complex refractive index

    '''
    # define constants
    eV = 1.602176634E-19          # eV to J (conversion)
    hbar = 1.0545718E-34          # J*s (plank's constan)
    
    
    w = 2*_np.pi*3E14/lam*hbar/eV  # conver from um to eV 
    
    return _np.sqrt(epsinf - wp**2/(w**2 + 1j*gamma*w))

def multi_oscillator(lam, oscilator_dict):
    '''
    Computes refractive index using a combination of oscillator models
    
    Parameters
    ----------
    lam  : ndarray or float
        Wavelength range (um)

    oscilator_dict   : dict
        Dictionary containing model parameters. Sorted as:
            {'lorentz': {'epsinf': value, 'wp': value, 'wn': value, 'gamma': value},
            'drude'  : {'epsinf': value, 'wp': value, 'gamma': value},
            'tauc-lorentz': {'A': value, 'C': value, 'E0': value, 'Eg': value},
            'gaussian': {'A': value, 'Br': value, 'E0': value} }

    Returns
    -------
    eps : ndarray (complex)
        Complex dielectric constant
    '''
    # create a dictionary with models and function names
    base_models = {
        'tauc-lorentz': tauc_lorentz,
        'gaussian': gaussian,
        'lorentz': lorentz,
        'drude': drude
    }

    eps = complex(0,0)
    for name, params in oscilator_dict.items():

        name = name.lower()
        if name not in base_models:
            print(f"Model '{name}' is not recognized. Skipping this model.")
            continue

        # Inspect model's signature to check required parameters
        sig = _inspect.signature(base_models[name])
        required_params = list(sig.parameters.keys())[1:]

        # Throw error if any required parameter is missing
        if set(params.keys()) != set(required_params):
            raise ValueError(f"{name} model requires parameters: {required_params}")
        
        eps += base_models[name](lam, **params)**2
    
    return _np.sqrt(eps)

def fit_to_oscillator(x, n_data, k_data, oscillator_dict, bounds=None, weights=None, x_units = 'um'):
    """
    Fit an oscillator parameters to tabulated refractive index data
    parameters:
    -----------
    x : array_like
        Independent variable array (e.g., wavelength in microns).
    n_data : array_like
        Real part of complex refractive index data to fit.
    k_data : array_like
        Imaginary part of complex refractive index data to fit.
    oscillator_dict : dict
        Dictionary containing oscillator model initial parameters. See `multi_oscillator` for format.
    bounds : dict, optional
        Bounds for oscillator parameters as (lower_bounds, upper_bounds).
    weights : array_like or scalar, optional
        Weights for fitting residuals. Can be a single scalar or a tuple of two arrays
        for real and imaginary parts.
    x_units : str, optional
        Units of the input x array. If None, assumes x is in microns.
    returns:
    --------
    DrudeParams
        Fitted Drude parameters as a named tuple (eps_inf, omega_p, gamma).
    res : OptimizeResult
        The optimization result returned by scipy.optimize.least_squares.
    """
    # convert x units to microns
    lam = _convert_units(x, x_units, to='um')
    lam = _np.asarray(lam, float)
        
    # Create a single array with real and imaginary parts
    y = _np.concatenate([_np.asarray(n_data, float), 
                        _np.asarray(k_data, float)])
    
    # Declare dictionary with standard bounds for each model
    bound_dict = {'drude':        {'epsinf': (0, 10), 'wp': (1E-2, 100), 'gamma': (1E-4, 10)},
                  'lorentz':      {'epsinf': (0, 10), 'wp': (1E-2, 100), 'wn': (1E-2, 10), 'gamma': (1E-4, 10)},
                  'tauc-lorentz': {'A': (0, 10), 'C': (0, 10), 'E0': (0, 10), 'Eg': (0, 10)},
                  'gaussian':     {'A': (0, 10), 'Br': (0, 10), 'E0': (0, 10)}}
    
    # create a dictionary with models and function names
    base_models = {
        'tauc-lorentz': tauc_lorentz,
        'gaussian': gaussian,
        'lorentz': lorentz,
        'drude': drude
    }

    if bounds is None:
        bounds = bound_dict
    elif not isinstance(bounds, dict):
        raise TypeError("Bounds must be provided as a dictionary.")

    # check that oscilator_dict has all required parameters and prepare initial values
    p0 = []
    lb , ub = [], []
    for name, params in oscillator_dict.items():

        name = name.lower()
        if name not in base_models:
            raise ValueError(f"Model '{name}' is not recognized.")

        if name not in bounds:
            raise ValueError(f"Bounds for model '{name}' are not recognized.")

        # Inspect model's signature to check required parameters
        sig = _inspect.signature(base_models[name])
        required_params = list(sig.parameters.keys())[1:]

        # Throw error if any required parameter is missing
        if set(params.keys()) != set(required_params):
            raise ValueError(f"{name} model requires parameters: {required_params}")
   
        model_bounds = bounds[name]
        for param_name in required_params:
            p0.append(params[param_name])          # append initial values
            lb.append(model_bounds[param_name][0]) # append lower bounds
            ub.append(model_bounds[param_name][1]) # append upper bounds

    p0 = _np.asarray(p0, float)
    p_bounds = (_np.asarray(lb), _np.asarray(ub))

    if weights is None:
        w = _np.ones_like(y)
    else:
        # weights can be (w_eps1, w_eps2) arrays, scalars, or a single scalar
        if _np.isscalar(weights): # if weight is a single number
            w = _np.ones_like(y) * float(weights)

        elif isinstance(weights, tuple) and len(weights) == 2: # if weight is a tuple with two scalar
            w1, w2 = weights
            w = _np.concatenate([_np.ones_like(n_data) * float(w1), 
                                _np.ones_like(k_data) * float(w2)])
        else:                                                  # if weight is a tuple with two arrays
            w1, w2 = weights
            w = _np.concatenate([_np.asarray(w1, float), _np.asarray(w2, float)])

    # ancillary function to construct oscillator dictionary from parameter array
    def construct_oscillator_dict(p):
        oscillator_iter = {}
        for name in oscillator_dict.keys():
            name = name.lower()
            sig = _inspect.signature(base_models[name])
            required_params = list(sig.parameters.keys())[1:]
            n_params = len(required_params)
            param_values = p[:n_params]
            p = p[n_params:]  # update p to remove used parameters
            oscillator_iter[name] = dict(zip(required_params, param_values))
        return oscillator_iter

    def resid(p):
        # construct dictionary with current parameters
        oscillator_iter = construct_oscillator_dict(p)

        # compute refractive index with current parameters
        n_complex = multi_oscillator(lam, oscillator_iter)
        r = _np.concatenate([n_complex.real - n_data, 
                            n_complex.imag - k_data])
        return r * w

    res = least_squares(resid, p0, bounds=p_bounds, method="trf")
    oscillator_iter = construct_oscillator_dict(res.x)
    return oscillator_iter, res

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
    eps_1, eps_1_isscalar = _ndarray_check(eps_1)
    eps_2, eps_2_isscalar = _ndarray_check(eps_2)

    # eps_1 is scalar, create a constant array of len(eps_2)
    if   eps_1_isscalar and not eps_2_isscalar:
        eps_1 = eps_1*_np.ones_like(eps_2)
        
    # eps_2 is scalar, create a constant array of len(eps_1)
    elif not eps_1_isscalar and eps_2_isscalar:
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

def eps_real_kkr(lam, eps_imag, eps_inf = 0, int_range = (0, _np.inf), cshift=1e-12):
    '''
    Computes real part of dielectric constant from its imaginary components 
    using Krammers-Kronig relation

    Parameters
    ----------
    lam: ndarray or float
         wavelength spectrum (in microns)
    
    eps_imag: ndarray, float or callable 
              imaginary component of refractive index (if ndarray, it must be same size as lam)
    
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
    lam, lam_isfloat = _ndarray_check(lam)
    cshift = complex(0, cshift)
    w_i = _convert_units(lam,'um', 'eV')

    if  isinstance(eps_imag, Callable):
        a, b = int_range # set integration range
        def integration_element(w_r):
            factor = lambda w: w / (w**2 - w_r**2 + cshift)
            real_int = lambda w: (eps_imag(w) * factor(w)).real
            imag_int = lambda w: (eps_imag(w) * factor(w)).imag
            total = quad(real_int, a,b)[0] + 1j*quad(imag_int, a,b)[0]
            return eps_inf + (2/_np.pi)*total
        
    elif isinstance(eps_imag, _np.ndarray) or isinstance(eps_imag,float):
        eps_imag = _ndarray_check(eps_imag)[0]
        assert lam.shape == eps_imag.shape, 'input arrays must be same length'
    
        def integration_element(w_r):
            factor = - w_i / (w_i**2 - w_r**2 + cshift) # integration domains are swaped, so a "-"" sign is added
            total = _np.trapz(eps_imag * factor, x=w_i)
            return eps_inf + (2/_np.pi)*total
    else:
        raise TypeError('Unknown type for eps_imag')
    
    eps_real = _np.real([integration_element(w_r) for w_r in w_i]).reshape(-1)
    
    if lam.shape == (1,):
        return float(eps_real[0])
    return float(eps_real[0]) if lam_isfloat else eps_real 
'''
    --------------------------------------------------------------------
                            Target functions
    --------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
#                                   Inorganic
# refractive index of SiO2 (quartz)
# SiO2 = lambda lam: get_nkfile(lam, 'sio2_Palik_Lemarchand2013', get_from_local_path = True)[0]
SiO2 = lambda lam: get_ri_info(lam, 'main', 'SiO2', 'Franta-25C')[0]

# refractive index of Fused silica
Silica = lambda lam: get_ri_info(lam, 'main', 'SiO2', 'Franta')[0]

# refractive index of CaCO3
CaCO3 = lambda lam: get_nkfile(lam, 'CaCO3_Palik', get_from_local_path = True)[0]

# refractive index of BaSO4
BaSO4 = lambda lam: get_nkfile(lam, 'BaSO4_Tong2022', get_from_local_path = True)[0]

# refractive index of BaF2
BaF2 = lambda lam: get_ri_info(lam, 'main', 'BaF2', 'Querry')[0]

# refractive index of TiO2
TiO2 = lambda lam: get_ri_info(lam,'main','TiO2','Siefke')[0]

# refractive index of BiVO4 monoclinic (a axis)
BiVO4_mono_a = lambda lam: get_nkfile(lam, 'BiVO4_a-c_Zhao2011', get_from_local_path = True)[0]

# refractive index of BiVO4 monoclinic (b axis)
BiVO4_mono_b = lambda lam: get_nkfile(lam, 'BiVO4_b_Zhao2011', get_from_local_path = True)[0]

# refractive index of BiVO4 monoclinic (c axis)
BiVO4_mono_c = lambda lam: get_nkfile(lam, 'BiVO4_a-c_Zhao2011', get_from_local_path = True)[0]

# average refractive index of BiVO4 monoclinic
BiVO4 = lambda lam: (BiVO4_mono_a(lam) + BiVO4_mono_b(lam) + BiVO4_mono_c(lam))/3

# refractive index of Cu2O
Cu2O = lambda lam: get_nkfile(lam, 'Cu2O_Malerba2011', get_from_local_path = True)[0]

# refractive index of ZnO
ZnO = lambda lam: get_ri_info(lam,'main','ZnO','Querry')[0]

# refractive index of MgO
MgO = lambda lam: get_nkfile(lam,'MgO_Palik', get_from_local_path = True)[0]

# refractive index of Alumina (AL2O3)
Al2O3 = lambda lam: get_ri_info(lam,'main','Al2O3','Querry-o')[0]

# refractive index of ZnS
ZnS = lambda lam: get_ri_info(lam,'main','ZnS','Querry')[0]

# refractive index of amorphous GeSbTe (GST)
GSTa = lambda lam: get_nkfile(lam, 'GSTa_Du2016', get_from_local_path = True)[0]

# refractive index of crystaline GeSbTe (GST)
GSTc = lambda lam: get_nkfile(lam, 'GSTc_Du2016', get_from_local_path = True)[0]

# refractive index of Monoclinic(cold) Vanadium Dioxide (VO2M)
# sputtered on SiO2 by default (film2)
VO2M = lambda lam, film = 2: get_nkfile(lam, 'VO2M_Wan2019(film%i)' % film, get_from_local_path = True)[0]

# refractive index of Rutile(hot) Vanadium Dioxide (VO2R)
# sputtered on SiO2 by default (film2)
VO2R = lambda lam, film = 2: get_nkfile(lam, 'VO2R_Wan2019(film%i)' % film, get_from_local_path = True)[0]

def VO2(lam,T, film=2 , Tphc = 73, WT = 3.1):
    '''
    Refractive index of temperatura dependent VO2.
    Reference: Wan, C. et al. Ann. Phys. 531, 1900188 (2019).

    Parameters
    ----------
    lam : ndarray
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
    eps_c = VO2M(lam,film)**2
    eps_h = VO2R(lam,film)**2
    
    eps = (1 - fv)*eps_c + fv*eps_h
    
    return _np.sqrt(eps)

# refractive index of Silicon
Si   = lambda lam: get_ri_info(lam, 'main', 'Si', 'Franta-300K')[0]

#------------------------------------------------------------------------------
#                                   Metals
# refractive index of Gold
gold = lambda lam: get_nkfile(lam, 'au_Olmon2012_evap', get_from_local_path = True)[0]

# refractive index of Silver
silver = lambda lam: get_nkfile(lam, 'ag_Ciesielski2017', get_from_local_path = True)[0]

# refractive index of Copper
Cu   = lambda lam: get_nkfile(lam, 'cu_Babar2015', get_from_local_path = True)[0]

# refractive index of Aluminium
Al   = lambda lam: get_nkfile(lam, 'al_Rakic1995', get_from_local_path = True)[0]

# refractive index of Magnesium
Mg   = lambda lam: get_ri_info(lam, 'main', 'Mg', 'Hagemann')[0]

#------------------------------------------------------------------------------
#                                   Polymers
# refractive index of HDPE
HDPE  = lambda lam: get_nkfile(lam, 'HDPE_Palik', get_from_local_path = True)[0]

# refractive index of HDPE
PDMS  = lambda lam: get_nkfile(lam, 'PDMS_Zhang2020_Querry1987', get_from_local_path = True)[0]

# refractive index of PMMA
PMMA = lambda lam: get_ri_info(lam,'organic','(C5H8O2)n - poly(methyl methacrylate)','Zhang-Tomson')[0]

# refractive index of PVDF-HFP
PVDF  = lambda lam: get_nkfile(lam, 'PVDF-HFP_Mandal2018', get_from_local_path = True)[0]

#------------------------------------------------------------------------------
#                                   Others
# refractive index of water
H2O  = lambda lam: get_nkfile(lam, 'h2o_Hale1973', get_from_local_path = True)[0]