import numpy as _np
from typing import Union as _Union, Optional as _Optional
from inspect import Signature as _Signature
from scipy.interpolate import CubicSpline as _CubicSpline
from typing import Tuple as _Tuple, Dict as _Dict
import pandas as _pd

# standard constants
e_charge = 1.602176634E-19      # C (elementary charge)
hbar = 1.0545718E-34            # J*s (plank's constan)
speed_of_light = 299792458      # m/s (speed of light)
kBoltzmann = 1.38064852E-23     # J/K (Boltzman constant)

def _as_1d_array(x, name, n_wavelengths=None, dtype=None):
    """Convert scalar/1D array-like input to a 1D ndarray.

    Accepts NumPy arrays, pandas Series, and array-like inputs exposing
    ``to_numpy()``. If ``n_wavelengths`` is provided, scalar inputs are
    broadcast and non-scalar inputs must match that length.
    """
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        arr = _np.asarray(x)

    if arr.ndim == 0:
        if n_wavelengths is None:
            arr = arr.reshape(1)
        else:
            n_wavelengths = int(n_wavelengths)
            if n_wavelengths <= 0:
                raise ValueError("n_wavelengths must be a positive integer.")
            scalar = arr.item()
            out_dtype = dtype if dtype is not None else arr.dtype
            arr = _np.full(n_wavelengths, scalar, dtype=out_dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array.")

    if n_wavelengths is not None and arr.shape != (int(n_wavelengths),):
        raise ValueError(f"{name} must be scalar or have shape (len(wavelength),).")

    if dtype is not None:
        arr = arr.astype(dtype)
    return arr

def _as_float_list(x, name, *, nonnegative=False):
    """Convert a scalar or 1D real array-like to a Python list of floats."""
    if x is None:
        return []
    return [float(v) for v in _as_1d_array(x, name, dtype=float)]

def _normalize_multilayer_inputs(wavelength, thickness, N_layers, *, N_above=1.0, N_below=1.0):
    """
    Normalize multilayer wave-optics inputs to the same scalar-or-spectrum style
    used elsewhere in the package.

    Returns
    -------
    wavelength : ndarray
        1D wavelength array [um].
    thicknesses : list[float]
        One thickness per finite layer [um].
    N_layers : list[ndarray]
        One complex spectral refractive-index array per finite layer.
    N_above_arr : ndarray
        Upper semi-infinite medium refractive index with shape (len(wavelength),).
    N_below_arr : ndarray
        Lower semi-infinite medium refractive index with shape (len(wavelength),).
    """
    wavelength_arr = _as_1d_array(wavelength, "wavelength", dtype=float)
    if thickness is None:
        dL = []
    else:
        thickness_arr = _np.asarray(thickness)
        if thickness_arr.ndim == 1 and thickness_arr.size == 0:
            dL = []
        else:
            dL = _as_float_list(thickness, "thickness", nonnegative=True)
    n_wavelengths = wavelength_arr.size

    if N_layers is None:
        layer_inputs = []
    elif _np.isscalar(N_layers):
        layer_inputs = [N_layers]
    else:
        arr = _np.asarray(N_layers)
        if arr.ndim == 0:
            layer_inputs = [N_layers]
        elif isinstance(N_layers, _np.ndarray):
            if arr.ndim == 1:
                layer_inputs = [N_layers]
            elif arr.ndim == 2:
                layer_inputs = [row for row in arr]
            else:
                raise TypeError(
                    "N_layers must be a scalar, a 1D spectral array, a 2D layer-by-wavelength array, or a list."
                )
        elif isinstance(N_layers, (list, tuple)):
            layer_inputs = list(N_layers)
        else:
            raise TypeError(
                "N_layers must be a scalar, a spectral array, or a list/tuple with one element per layer."
            )

    if len(layer_inputs) != len(dL):
        raise ValueError("len(N_layers) must be equal to len(thickness).")

    nL = [
        _as_1d_array(layer_ni, f"N_layers[{i}]", n_wavelengths=n_wavelengths, dtype=complex)
        for i, layer_ni in enumerate(layer_inputs)
    ]
    n0 = _as_1d_array(N_above, "N_above", n_wavelengths=n_wavelengths, dtype=complex)
    nS = _as_1d_array(N_below, "N_below", n_wavelengths=n_wavelengths, dtype=complex)

    return wavelength_arr, dL, nL, n0, nS

# a function to convert units in electrodynamics
def convert_units(x, x_in, to):
    '''
    Convert units of a variable. Accepted units for conversion are:
        nanometers              : 'nm'
        micrometers             : 'um'
        recriprocal centimeters : 'cm^-1'
        frequency               : 'Hz'
        angular frequency       : 'rad/s'
        electron voltz          : 'eV'

    Parameters
    ----------
    x : ndarray
        list of values to convert.
    x_in : string
        units of the input variable.
    to : string
        conversion units.

    Returns
    -------
    ndarray
        coverted list of values.

    '''
    
    eV = 1.602176634E-19      # C (elementary charge)
    hbar = 1.0545718E-34      # J*s/rad (red. plank's constan)
    PI = 3.141592653589793
    c0 = speed_of_light       # m/s (speed of light)
    hbar = hbar/eV            # eV*s/rad (red. plank's constan)
    h = 2*PI*hbar             # eV/Hz (plank's constan)
    
    unit_dict = ['nm', 'um', 'cm^-1', 'Hz', 'rad/s', 'eV']
    
    assert x_in in unit_dict, 'Unkown unit: ' + x_in
    assert to in unit_dict, 'Unkown unit: ' + to
    
    unit_table = {
        ('nm','nm')       : x, 
        ('nm','um')       : x*1E-3, 
        ('nm','cm^-1')    : 1/x*1E7, 
        ('nm','Hz')       : c0/x*1E9,
        ('nm','rad/s')    : 2*PI*c0/x*1E9,
        ('nm','eV')       : h*c0/x*1E9, 
        
        ('um','nm')       : x*1E3, 
        ('um','um')       : x, 
        ('um','cm^-1')    : 1/x*1E4, 
        ('um','Hz')       : c0/x*1E6,
        ('um','rad/s')    : 2*PI*c0/x*1E6,
        ('um','eV')       : h*c0/x*1E6, 
        
        ('cm^-1','nm')    : 1/x*1E7, 
        ('cm^-1','um')    : 1/x*1E4,  
        ('cm^-1','cm^-1') : x, 
        ('cm^-1','Hz')    : x*c0*1E2,
        ('cm^-1','rad/s') : x*2*PI*c0*1E2,
        ('cm^-1','eV')    : x*h*c0*1E2, 
        
        ('Hz','nm')       : c0*1E9/x, 
        ('Hz','um')       : c0*1E6/x,
        ('Hz','cm^-1')    : x/(c0*1E2), 
        ('Hz','Hz')       : x,
        ('Hz','rad/s')    : 2*PI*x,
        ('Hz','eV')       : h*x, 
        
        ('rad/s','nm')    : 2*PI*c0/x*1E9, 
        ('rad/s','um')    : 2*PI*c0/x*1E6,
        ('rad/s','cm^-1') : x/(2*PI*c0*1E2), 
        ('rad/s','Hz')    : x/(2*PI),
        ('rad/s','rad/s') : x,
        ('rad/s','eV')    : hbar*x, 
        
        ('eV','nm')       : h*c0/x*1E9, 
        ('eV','um')       : h*c0/x*1E6,
        ('eV','cm^-1')    : x/(h*c0*1E2), 
        ('eV','Hz')       : x/h,
        ('eV','rad/s')    : x/hbar,
        ('eV','eV')       : x, 
        
        }
    
    return unit_table[(x_in, to)]

def _local_to_global_angles(theta_i, phi_i, beta_tilt, phi_tilt, restrict_to_upper_hemisphere=False):
    """
    Converts local spherical angles (theta_i, phi_i), defined relative to the normal of a tilted surface,
    into global spherical angles (theta, phi) defined with respect to the global vertical (z-axis).

    Parameters:
    -----------
    theta_i : float or ndarray
        Local zenith angle in radians (relative to the tilted surface normal).
    phi_i : float or ndarray
        Local azimuth angle in radians (relative to the tilted surface normal).
    beta : float
        Tilt angle of the surface with respect to the global vertical (radians).
    phi_tilt : float
        Azimuth direction of the surface tilt in the global frame (radians).
    restrict_to_upper_hemisphere : bool
        If True, any direction that ends up in the lower global hemisphere (theta > π/2) will be reflected upward.

    Returns:
    --------
    theta : float or ndarray
        Global zenith angle in radians (relative to vertical).
    phi : float or ndarray
        Global azimuth angle in radians (0 to 2π).
    """
    import numpy as np

    original_shape = _np.shape(theta_i)
    theta_i = _np.ravel(theta_i)
    phi_i = _np.ravel(phi_i)

    # Tilted normal vector in global coordinates
    n_local = _np.array([
        _np.sin(beta_tilt) * _np.cos(phi_tilt),
        _np.sin(beta_tilt) * _np.sin(phi_tilt),
        _np.cos(beta_tilt)
    ])
    
    # Rotation axis: cross product between global z and tilted normal
    axis = _np.cross([0, 0, 1], n_local)
    axis_norm = _np.linalg.norm(axis)

    if axis_norm < 1e-8:
        # No tilt: return inputs unchanged
        theta = theta_i.reshape(original_shape)
        phi = phi_i.reshape(original_shape)
        return theta.item() if theta.size == 1 else theta, phi.item() if phi.size == 1 else phi

    axis = axis / axis_norm  # normalize

    # Rodrigues' rotation matrix for rotation around arbitrary axis
    cos_b = _np.cos(beta_tilt)
    sin_b = _np.sin(beta_tilt)
    ux, uy, uz = axis

    R = _np.array([
        [cos_b + ux**2 * (1 - cos_b),     ux*uy*(1 - cos_b) - uz*sin_b,  ux*uz*(1 - cos_b) + uy*sin_b],
        [uy*ux*(1 - cos_b) + uz*sin_b,    cos_b + uy**2 * (1 - cos_b),   uy*uz*(1 - cos_b) - ux*sin_b],
        [uz*ux*(1 - cos_b) - uy*sin_b,    uz*uy*(1 - cos_b) + ux*sin_b,  cos_b + uz**2 * (1 - cos_b)]
    ])

    # Convert local spherical to Cartesian coordinates
    x = _np.sin(theta_i) * _np.cos(phi_i)
    y = _np.sin(theta_i) * _np.sin(phi_i)
    z = _np.cos(theta_i)
    v_local = _np.stack([x, y, z], axis=0)  # shape: (3, N)

    # Rotate to global coordinates
    v_global = R @ v_local

    # Optional: reflect any downward-pointing vectors if restricting to upper hemisphere
    if restrict_to_upper_hemisphere:
        below = v_global[2] < 0
        v_global[:, below] *= -1

    # Convert to global spherical coordinates
    xg, yg, zg = v_global[0], v_global[1], v_global[2]
    theta = _np.arccos(_np.clip(zg, -1, 1))
    phi = _np.mod(_np.arctan2(yg, xg), 2 * _np.pi)

    # Restore original shape
    theta = theta.reshape(original_shape)
    phi = phi.reshape(original_shape)

    if original_shape == ():
        return theta.item(), phi.item()
    return theta, phi

def _check_mie_inputs(wavelength=None, N_host=None, Np_shells=None, D=None, *, size_dist=None):
    """
    Validate and normalize inputs for Mie / multilayer-sphere calculations.

    Parameters
    ----------
    wavelength : float or (nλ,) array-like of float, optional
        Wavelength(s) in micrometers (µm). If array-like, must be 1D with wavelength > 0.
        If omitted (None), spectral refractive indices cannot be used (only scalars allowed).
    N_host : complex or (nλ,) array-like of complex, optional
        Host refractive index. If array-like, length must equal len(wavelength).
        If scalar, it is broadcast to (nλ,). If wavelength is None and N_host is array-like, error.
    Np_shells : scalar complex, (nλ,) array-like of complex, list/tuple of those,
                or (n_layers, nλ) ndarray, optional
        Refractive index for each shell layer.
        Accepted forms:
          - scalar (single-layer, constant with λ)
          - 1D array (single-layer spectrum, length = len(wavelength))
          - list/tuple of scalars/1D arrays (one per layer; arrays must match len(wavelength))
          - 2D ndarray shaped (n_layers, nλ)
        If arrays are provided but wavelength is None, error.
    D : float, array-like, or list of those, optional
        Outer diameter(s) per layer (µm). Semantics:
          - float: single-layer, monodisperse
          - 1D array: single-layer, polydisperse (size distribution), or
            multilayer monodisperse when `Np_shells` clearly defines >1 layer
          - list of floats: multilayer, monodisperse
          - list of 1D arrays: multilayer, polydisperse (all arrays must be same length)
          - 2D array: multilayer, polydisperse with shape (n_layers, n_bins)
        For multilayer monodisperse (list of floats): strictly increasing.
        For multilayer polydisperse (list of arrays): element-wise strictly increasing across layers.
    size_dist : None or 1D array-like of float, optional (default None)
        Size-distribution weights. If omitted for polydisperse `D`, a uniform
        number distribution is assumed and normalized automatically.

    Returns
    -------
    wavelength_out : (nλ,) ndarray of float or None
    N_host_out : (nλ,) ndarray of complex or None
    Np_out : (n_layers, nλ) ndarray of complex or None
        Layers ordered inner→outer. If D is provided, rows are reordered accordingly.
    D_out : list[ndarray]
        One ndarray per layer. For monodisperse layers, each entry is a length-1 array.
    size_dist_out : (n_bins,) ndarray of float
        Normalized size-distribution vector. For monodisperse, array([1.0]).

    Notes
    -----
    - If both D and Np_shells are provided, their number of layers must match.
    - For polydisperse `D`, omitted `size_dist` defaults to uniform weights.
    """
    # ---- wavelength ----
    if wavelength is None:
        wavelength_out = None
    else:
        wavelength_arr = _np.asarray(wavelength, dtype=float).ravel()
        if wavelength_arr.ndim != 1 or wavelength_arr.size == 0:
            raise ValueError("wavelength must be a 1D array (non-empty) or a scalar.")
        if not _np.all(_np.isfinite(wavelength_arr)) or _np.any(wavelength_arr <= 0):
            raise ValueError("All wavelengths in wavelength must be finite and > 0 (µm).")
        wavelength_out = wavelength_arr

    n_wavelengths = None if wavelength_out is None else wavelength_out.size

    def _infer_layer_count(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return len(x)
        arr = _np.asarray(x)
        if arr.ndim == 2:
            return int(arr.shape[0])
        return 1

    n_layers_hint = _infer_layer_count(Np_shells)

    # ---- D (diameters) → list[ndarray]
    D_out = None
    n_bins = None  # number of size bins if polydisperse

    if D is not None:
        def _as_1d_array_positive(x):
            return _as_1d_array(x, "D", dtype=float)

        def _validate_monodisperse_layers(d_layers):
            mono_vals = _np.array([a.item() for a in d_layers], dtype=float)
            if _np.any(_np.diff(mono_vals) <= 0):
                raise ValueError(
                    "For multilayer monodisperse, D must be strictly increasing from inner to outer layer."
                )

        def _validate_polydisperse_layers(d_layers):
            lengths = _np.array([a.size for a in d_layers], dtype=int)
            if not _np.all(lengths == lengths[0]):
                raise ValueError(
                    "For multilayer polydisperse, all layer arrays in D must have the same length."
                )
            D_stack = _np.vstack(d_layers)
            if _np.any(_np.diff(D_stack, axis=0) <= 0):
                raise ValueError(
                    "For multilayer polydisperse, diameters must be element-wise strictly increasing across layers."
                )
            return int(lengths[0])
        
        # single-layer monodisperse
        if _np.isscalar(D) or (isinstance(D, _np.ndarray) and _np.asarray(D).ndim == 0):
            D_out = [_np.array([float(D)], dtype=float)]

        elif isinstance(D, _np.ndarray):
            arr = _np.asarray(D)
            if arr.ndim == 1:
                D_arr = _as_1d_array_positive(D)
                if n_layers_hint is not None and n_layers_hint > 1:
                    if D_arr.size != n_layers_hint:
                        raise ValueError(
                            "For multilayer inputs, a 1D array D must provide exactly one diameter per layer."
                        )
                    D_out = [_np.array([float(di)], dtype=float) for di in D_arr]
                    _validate_monodisperse_layers(D_out)
                else:
                    D_out = [D_arr]
            elif arr.ndim == 2:
                D_arr = _np.asarray(D, dtype=float)
                if D_arr.size == 0:
                    raise ValueError("D arrays must be non-empty.")
                if n_layers_hint is not None and D_arr.shape[0] != n_layers_hint and D_arr.shape[1] == n_layers_hint:
                    D_arr = D_arr.T
                if n_layers_hint is not None and D_arr.shape[0] != n_layers_hint:
                    raise ValueError(
                        "For multilayer inputs, a 2D array D must have shape (n_layers, n_bins)."
                    )
                D_out = [_as_1d_array_positive(row) for row in D_arr]
                n_bins = _validate_polydisperse_layers(D_out)
            else:
                raise TypeError("D must be a scalar, a 1D/2D array, or a list/tuple of layer diameters.")

        # multilayer sphere
        elif isinstance(D, (list, tuple)):
            if len(D) == 0:
                raise ValueError("D cannot be an empty list.")
            
            D_list = []
            for d in D:
                # monodisperse layer (scalar)
                if _np.isscalar(d) or (isinstance(d, _np.ndarray) and _np.asarray(d).ndim == 0):
                    D_list.append(_np.array([float(d)], dtype=float))
                
                # polydisperse layer (1D array)
                else:
                    arr = _as_1d_array_positive(d)
                    D_list.append(arr)

            lengths = _np.array([a.size for a in D_list], dtype=int)
            if _np.any(lengths > 1):
                n_bins = _validate_polydisperse_layers(D_list)
            else:
                _validate_monodisperse_layers(D_list)

            D_out = D_list

        else:
            raise TypeError("D must be a scalar, a 1D/2D array, or a list/tuple of layer diameters.")

        if n_bins is None:
            n_bins = int(max(a.size for a in D_out))

        # Enforce either all mono or all share n_bins>1
        if not (all(a.size == 1 for a in D_out) or all(a.size == n_bins for a in D_out)):
            raise ValueError("All layers must be monodisperse (length=1) or all polydisperse with a common length.")

    # ---- Np_shells (layers) → (n_layers, nλ)
    Np_out = None
    if Np_shells is not None:
        def to_layer_array(x):
            xa = _np.asarray(x)
            if xa.ndim == 0:  # scalar
                if n_wavelengths is None:
                    return _np.array([complex(xa)], dtype=complex)
                return _np.full(n_wavelengths, complex(xa), dtype=complex)
            arr = _np.asarray(xa, dtype=complex).ravel()
            if wavelength_out is None:
                print('No wavelength provided, but spectral Np_shells given. Ignoring spectral shape.')
            elif arr.size != n_wavelengths:
                raise ValueError(f"A spectral layer has length {arr.size}, expected len(wavelength)={n_wavelengths}.")
            return arr

        if isinstance(Np_shells, (list, tuple)):
            if len(Np_shells) == 0:
                raise ValueError("Np_shells list cannot be empty.")
            layers = [to_layer_array(x) for x in Np_shells]
            if n_wavelengths is not None and any(arr.size != n_wavelengths for arr in layers):
                raise ValueError("All spectral layers in Np_shells must have length len(wavelength).")
            Np_out = _np.vstack([arr.reshape(1, -1) for arr in layers]).astype(complex)
        else:
            arr = _np.asarray(Np_shells)
            if arr.ndim == 0:
                Np_out = to_layer_array(arr).reshape(1, -1)
            elif arr.ndim == 1:
                Np_out = to_layer_array(arr).reshape(1, -1)
            elif arr.ndim == 2:
                if wavelength_out is None:
                    raise ValueError("2D Np_shells provided but wavelength is None. Provide wavelength.")
                if arr.shape[1] != n_wavelengths:
                    raise ValueError(f"Np_shells second dimension must equal len(wavelength)={n_wavelengths}.")
                if not _np.all(_np.isfinite(arr)):
                    raise ValueError("Np_shells contains non-finite values.")
                Np_out = arr.astype(complex)
            else:
                raise TypeError("Np_shells must be scalar, 1D array, list/tuple of scalars/1D arrays, or 2D array.")

    # ---- Cross-check layers vs diameters ----
    if (Np_out is not None) and (D_out is not None):
        if Np_out.shape[0] != len(D_out):
            raise ValueError(
                f"Number of layers mismatch: len(D)={len(D_out)} but Np_shells has {Np_out.shape[0]} layer(s)."
            )

    # ---- N_host ----
    N_host_out = None
    if N_host is not None:
        if _np.isscalar(N_host):
            if n_wavelengths is None:
                N_host_out = _np.array([complex(N_host)], dtype=complex)
            else:
                N_host_out = _np.full(n_wavelengths, complex(N_host), dtype=complex)
        else:
            if wavelength_out is None:
                raise ValueError("Spectral N_host provided but wavelength is None. Provide wavelength.")
            arr = _np.asarray(N_host, dtype=complex).ravel()
            if arr.size != n_wavelengths:
                raise ValueError(f"N_host length must equal len(wavelength)={n_wavelengths}.")
            N_host_out = arr

    # ---- size_dist (normalize & validate) ----
    if D_out is None:
        # Without D, default to None
        size_dist_out = None
    else:
        is_mono = all(a.size == 1 for a in D_out)
        if size_dist is None:
            if is_mono:
                size_dist_out = None
            else:
                n_bins = D_out[0].size
                print(
                    "size_dist is None for polydisperse D. "
                    "Assuming a uniform number distribution."
                )
                size_dist_out = _np.full(n_bins, 1.0 / n_bins, dtype=float)
        else:
            sd = _np.asarray(size_dist, dtype=float).ravel()
            if not _np.all(_np.isfinite(sd)) or _np.any(sd < 0) or sd.size == 0:
                raise ValueError("size_dist must contain non-negative finite values.")
            if is_mono:
                if sd.size != 1:
                    raise ValueError("For monodisperse D, size_dist must be length 1.")
                s = sd.sum()
                if s <= 0:
                    raise ValueError("size_dist sum must be > 0.")
                size_dist_out = sd / s
            else:
                n_bins = D_out[0].size  # all layers share same length if polydisperse
                if sd.size != n_bins:
                    raise ValueError(f"size_dist length {sd.size} must match number of size bins {n_bins}.")
                s = sd.sum()
                if s <= 0:
                    raise ValueError("size_dist sum must be > 0.")
                size_dist_out = sd / s

    # ---- Final NaN/Inf guards ----
    for name, arr in [("wavelength", wavelength_out), ("N_host", N_host_out), ("Np_shells", Np_out)]:
        if arr is None:
            continue
        if not _np.all(_np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")

    if D_out is not None:
        for a in D_out:
            if not _np.all(_np.isfinite(a)):
                raise ValueError("D contains non-finite values.")

    return wavelength_out, N_host_out, Np_out, D_out, size_dist_out

def _check_theta(theta: _Optional[_Union[float, _np.ndarray]],
                 n_theta: int = 181) -> _np.ndarray:
    """
    Validate and format the scattering angle array (theta).

    Parameters
    ----------
    theta : float or ndarray or None
        Scattering angle(s) in radians. If None, generates a dense grid from 0 to π.
    n_theta : int, optional
        Number of points in the angular grid if `theta` is None.

    Returns
    -------
    theta : _np.ndarray
        1D array of angles in radians, shape (n_theta,).
    """
    if theta is None:
        theta = _np.linspace(0.0, 2*_np.pi, max(int(n_theta), 5))
    
    elif _np.isscalar(theta):
        theta = _np.array([float(theta)])
    
    else:
        theta = _np.asarray(theta, dtype=float).ravel()

    return theta

# decorator to hide function signature
def _hide_signature(func):
    try:
        func.__signature__ = _Signature()
    except Exception:
        pass
    return func

def _warn_extrapolation(lam_arr, lo, hi, label="", quantity=""):
    lam_min = float(_np.min(lam_arr))
    lam_max = float(_np.max(lam_arr))
    if lam_min < lo and lam_max > hi:
        print(
            f"Extrapolating {label} {quantity} (requested {lam_min:.3f}–{lam_max:.3f} µm; "
            f"data {lo:.3f}–{hi:.3f} µm)"
            )
        
    else:
        if lam_min < lo:
            print(
                f"Extrapolating {label} {quantity} below tabulated range "
                f"(requested min {lam_min:.3f} µm; data starts {lo:.3f} µm)",
               )
        if lam_max > hi:
            print(
                f"Extrapolating {label} {quantity} above tabulated range "
                f"(requested max {lam_max:.3f} µm; data ends {hi:.3f} µm)",
            )

def _check_aoi(incidence_angle):
    aoi_max = np.pi/2
    if incidence_angle > aoi_max:
        raise ValueError("aoi > pi/2")

def rt_style_mapper(sample: _pd.DataFrame,
                linestyles: _Dict = {"tot": "-", "spec": ":", "dif": "--"},
                colors: _Dict = {"R": "r", "T": "b", "A": "k"}) -> str:
    """
    Generate matplotlib linestyles and legend labels for RT DataFrame results.
    E.g. {"Rtot": "-r", "Tspec": ":b", "Rdif": "--r"}.
    
    Parameters
    ----------
    sample : pandas.DataFrame
        DataFrame with columns named like Rtot, Tspec, Rdif, Atot, etc.
    
    linestyles : Dict, optional
        Mapping of line type keywords to matplotlib line styles.
        Defaults to {"tot": "-", "spec": ":", "dif": "--"}.
    
    colors : Dict, optional
        Mapping of measurement type keywords to colors.
        Defaults to {"R": "r", "T": "b", "A": "k"}.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping each column name to a matplotlib style string.
        E.g. {"Rtot": "-r", "Tspec": ":b", "Rdif": "--r"}
    
    Notes
    -----
    - The function looks for keywords in column names to determine line style
      and color. It is case-insensitive.
    
    - If a column name does not match any known keywords, it defaults to
      a solid black line ("-k").
    """
    
    style = {}
    label = {}
    # Case-insensitive matching
    for col_name in sample.columns:
        color = colors.get(col_name[0], "k")  # default black if not R/T
        rt_type = col_name[0]  # first letter indicates R, T, or A
        
        found = False
        for key, ls in linestyles.items():
            # match line style keywords at end of column name
            if col_name.lower().endswith(key):
                style[col_name] = ls + color
                label[col_name] = f"${rt_type}_\\mathrm{{{key}}}$"
                found = True
                break

        # Fallback solid line if no style keyword matched
        if not found:
            style[col_name] = "-" + color
            
    return style, label

def detect_spectral_spikes(
    x: _np.ndarray,
    y: _np.ndarray,
    *,
    k: float = 4.0,                     # robust threshold: flag if |(mL*mR) - med| / MAD > k  AND (mL*mR) < 0
    min_slope: float | None = None,     # ignore “flat” regions where both slopes are tiny; if None -> 25th pct of |m|
    dilate: int = 0,                    # expand the mask by this many neighbors on each side
    max_frac_removed: float = 0.25,     # if more than this fraction would be removed, abort cleaning (return original)
    return_mask: bool = False,
) -> _np.ndarray | _Tuple[_np.ndarray, _np.ndarray]:
    """
    Clean 1D spectra by removing single-point spikes identified via the
    product of adjacent slopes.

    A point i (1..n-2) is flagged if the slopes on its left and right
        segments are (a) of opposite sign and (b) the slope-product is an
        outlier relative to its robust local scale.

        Parameters
        ----------
        x, y : array_like
        Abscissa (strictly increasing after internal sorting) and ordinate.
    k : float, default 4.0
        Robust z-score threshold using median/MAD on slope products. Larger
        k is stricter (fewer flags).
    min_slope : float or None, default None
        Magnitude gate for slopes; both |m_L| and |m_R| must exceed this.
        If None, it is set to the 25th percentile of |m| (conservative).
        This prevents flagging tiny zig-zags in flat regions.
    dilate : int, default 0
        Binary dilation radius for the mask (grabs immediate neighbors of
        a one-point spike).
    max_frac_removed : float, default 0.25
        Safety limit: if more than this fraction of points would be removed,
        the function returns the original y unchanged.
    return_mask : bool, default False
        If True, return (y_clean, mask). Otherwise just y_clean.

    Returns
    -------
    y_clean : ndarray
        Cleaned series using CubicSpline through non-flagged points.
    mask : ndarray of bool (optional)
        True where points were flagged as spikes.

    Notes
    -----
    - Uses *global* robust stats on m_L*m_R; you asked for a simple,
      global criterion rather than windowed/local analysis.
    - Requires at least 4 unique x-values after sorting/condensing.
    - If too few good points remain for a spline, the original y is returned.
    """
    x = _np.asarray(x, float)
    y = _np.asarray(y, float)
    n = y.size

    if n < 4:
        return (y.copy(), _np.zeros(n, bool)) if return_mask else y.copy()

    # ---- 1) sort by x and condense exact duplicates (average y at same x) ----
    sorted = False
    if not _np.all(_np.diff(x) > 0):
        order = _np.argsort(x)
        x, y = x[order], y[order]
        sorted = True

    xu, idx, counts = _np.unique(x, return_index=True, return_counts=True)
    if xu.size != x.size:
        # average y over runs of identical x
        y = _np.add.reduceat(y, idx) / counts
        x = xu
        n = x.size
        if n < 4:
            return (y.copy(), _np.zeros(n, bool)) if return_mask else y.copy()
    # ---- 2) adjacent slopes (non-uniform x allowed) ----
    dx = _np.diff(x)
    dy = _np.diff(y)
    # guard against zero-division (shouldn't happen after condensing, but be safe)
    eps = 1e-15
    m = dy / (dx + eps)            # length n-1
    mL, mR = m[:-1], m[1:]         # each length n-2

    # ---- 3) robust outlier score on slope-product ----
    prod = mL * mR                 # negative for sign flip; large |.| for sharp turns
    med = _np.median(prod)
    mad = _np.median(_np.abs(prod - med))
    scale = 1.4826 * (mad if mad > 0 else _np.median(_np.abs(prod)) + eps)
    z = _np.abs(prod - med) / (scale + eps)

    # opposite directions required (up/down or down/up)
    sign_flip = prod < 0.0

    # magnitude gate to avoid tiny wiggles
    if min_slope is None:
        min_slope = _np.percentile(_np.abs(m), 25.0)
    mag_ok = (_np.abs(mL) > min_slope) & (_np.abs(mR) > min_slope)
    core = sign_flip & mag_ok & (z > k)   # main condition on internal vertices

    # ---- 4) build full-length mask and optional dilation ----
    mask = _np.zeros(n, dtype=bool)
    mask[1:-1] = core

    if dilate > 0 and core.any():
        for r in range(1, dilate + 1):
            mask[:-r] |= mask[r:]
            mask[r:]  |= mask[:-r]

    # ---- 5) safety limits & spline repair ----
    n_bad = int(mask.sum())
    if n_bad == 0 or n_bad / float(n) > max_frac_removed or (n - n_bad) < 4:
        # nothing to do or would remove too much / too few points left
        return (y, mask) if return_mask else y

    xs = x[~mask]
    ys = y[~mask]

    try:
        cs = _CubicSpline(xs, ys, bc_type="not-a-knot")
        y_clean = cs(x)
    except Exception:
        # fallback: leave unchanged if spline fails for any reason
        y_clean = y.copy()

    if sorted:
        # restore original order
        y_final = _np.empty_like(y_clean)
        y_final[order] = y_clean
        y_clean = y_final
        if return_mask:
            mask_final = _np.empty_like(mask)
            mask_final[order] = mask
            mask = mask_final

    return (y_clean, mask) if return_mask else y_clean
