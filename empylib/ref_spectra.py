# -*- coding: utf-8 -*-
"""
This library contains reference spectra:
    AM1.5
    Plank's distribution
    Atmospheric Transmittance

Created on Fri Jan 21 16:05:48 2022

@author: PanxoPanza
"""

import numpy as _np 
import empylib as _em
from .utils import _ndarray_check, _local_to_global_angles
from pathlib import Path as _Path
from typing import Optional as _Optional, Tuple as _Tuple, Sequence as _Sequence, List as _List, Dict as _Dict, Any as _Any, Union as _Union

__all__ = ['read_spectrafile', 
           'AM15', 
           'T_atmosphere', 
           'T_atmosphere_hemi', 
           'Bplanck', 
           'yCIE_lum', 
           'spectral_average', 
           'plot_spectra']

# Global cache for loaded files and interpolators
_file_cache = {}

def read_spectrafile(lam, MaterialName, get_from_local_path=False, return_data=False):
    """
    Reads a text file and returns an interpolated 1D NumPy array 
    for the specified material's spectral data.

    Parameters
    ----------
    lam : float or ndarray
        Wavelengths (in µm) to interpolate.
    MaterialName : str
        Name of the file (with extension).
    get_from_local_path : bool, optional
        If True, reads from the script's folder instead of the working directory.
    return_data : bool, optional
        If True, returns both the interpolated values and the full data array.

    Returns
    -------
    out : ndarray
        Interpolated values at requested wavelengths.
    data : ndarray (optional)
        Original tabulated data.
    """

    lam = _np.atleast_1d(lam)

    # Resolve path
    if get_from_local_path:
        caller_directory = _Path(__file__).parent / 'spectra_data'
    else:
        caller_directory = _Path.cwd()

    file_path = str(caller_directory / MaterialName)

    # Check and load file from cache or disk
    if file_path not in _file_cache:
        # Load file
        assert _Path(file_path).exists(), f"File not found: {file_path}"
        data = _np.genfromtxt(file_path)

        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Invalid file format; expected two columns.")

        # Save to cache
        _file_cache[file_path] = data
    else:
        data = _file_cache[file_path]

    # Interpolate
    out = _np.interp(lam, data[:, 0], data[:, 1], left=0, right=0)

    if return_data:
        return out, data
    return out

def AM15(lam,spectra_type='global'):
    '''
    AM1.5 spectra

    Parameters
    ----------
    lam : 1D float array (or scalar)
        wavelength in um.

    Returns
    -------
    Interpolated AM1.5 spectra

    '''
    # interpolate values according to lam spectra
    lam = lam*1E3 # change units to nm
    
    if spectra_type == 'global':
        Isun = read_spectrafile(lam,'AM15_Global.txt', True)
    elif spectra_type == 'direct':
        Isun = read_spectrafile(lam,'AM15_Direct.txt', True)
    
    # keep only positive values
    Isun = _np.clip(Isun, 0, None)
    
    return Isun*1E3  # spectra in W/m2 um

def T_atmosphere(lam):
    '''
    Spectral transmissivity of the atmosphere for an horizontal surface 
    at normal incidence. Data taken from:
        IR Transmission Spectra, Gemini Observatory Kernel Description. 
        http://www.gemini.edu/?q/node/10789, accessed Sep 27, 2018.

    Parameters
    ----------
    lam : 1D float array (or scalar)
        wavelength in um.

    Returns
    -------
    Interpolated Transmissivity of the atmosphere

    '''
    # interpolate values according to lam spectra
    T_atm =  read_spectrafile(lam,'T_atmosphere.txt', True)
    
    # keep only positive values
    T_atm = _np.clip(T_atm, 0, None)

    return T_atm

def T_atmosphere_hemi(lam, beta_tilt=0):
    """
    Computes the hemispherical atmospheric transmittance spectrum over a surface tilted at a given angle.
    This function integrates the directional atmospheric transmittance over a hemisphere centered 
    on a surface tilted by `beta_tilt` degrees. It accounts for the angular dependence of 
    radiative path length through the atmosphere, assuming unity transmittance at grazing angles 
    (zenith > 90°).

    Parameters
    ----------
    lam : 1D ndarray
        Wavelengths in micrometers [μm].
    beta_tilt : float, optional
        Tilt angle of the surface in degrees with respect to the vertical (default is 0°).

    Returns
    -------
    T_hemi : 1D ndarray
        Hemispherical atmospheric transmittance spectrum corresponding to each wavelength in `lam`.

    Notes
    -----
    - The integration is performed over solid angles using a weighted cosine projection.
    - Transmittance is assumed to be 1 for zenith angles greater than 90°, consistent with
      complete atmospheric opacity at grazing incidence.
    - The output is shifted to ensure minimum transmittance starts from 0 for normalization purposes.
    """

    beta = _np.radians(beta_tilt)

    # Angular grid
    theta_i = _np.linspace(0, _np.pi, 30)
    phi_i = _np.linspace(0, 2 * _np.pi, 30)
    tt, pp = _np.meshgrid(theta_i, phi_i, indexing='ij')  # shape: (T, P)

    theta, phi = _local_to_global_angles(tt, pp, beta, phi_tilt=0)  # shape: (T, P)

    # Compute angular weights
    weight = _np.cos(tt) * _np.sin(tt)  # shape: (T, P)

    # Flatten angles
    theta_flat = theta.ravel()  # shape: (N,)
    weight_flat = weight.ravel()  # shape: (N,)

    # Transmission mask
    mask = theta_flat < _np.pi / 2
    cos_theta = _np.cos(theta_flat[mask])  # shape: (M,)

    # T_atmosphere for all wavelengths
    T_vec = T_atmosphere(lam)[:, None]  # shape: (L,1)

    # Compute directional emissivity: (L, N)
    trans = _np.ones((len(lam), len(theta_flat)))  # shape: (L, N)
    trans[:, mask] = T_vec**(1 / cos_theta)  # broadcasting over wavelengths

    # Integrate over angles
    integrand = trans * weight_flat  # shape: (L, N)
    dphi = phi_i[1] - phi_i[0]
    dtheta = theta_i[1] - theta_i[0]

    # Sum over all angles
    T_hemi = _np.sum(integrand, axis=1) * dphi * dtheta / _np.pi # shape: (L,)

    # Adjust to enssure 0 < T < 1
    T_hemi = T_hemi - _np.min(T_hemi)
    T_hemi = T_hemi/max(_np.max(T_hemi), 1.0)

    return T_hemi

def Bplanck(lam, T, unit='wavelength'):
    """
    Spectral Planck black-body distribution (radiance).
    
    Parameters
    ----------
    lam : array_like
        Wavelength(s) in microns (um) if unit='wavelength';
        still passed in microns when unit='frequency' (frequency computed from lam).
    T : float
        Temperature in K.
    unit : {'wavelength','frequency'}
        Output units:
          - 'wavelength': W / (m^2 · um · sr)
          - 'frequency' : W / (m^2 · Hz · sr)

    Returns
    -------
    Ibb : ndarray
        Spectral radiance in the units indicated above.
    """
    # --- constants (float64) ---
    c0   = float(_em.speed_of_light)  # m/s
    hbar = float(_em.hbar)            # J·s/rad
    h    = 2.0*_np.pi*hbar             # J·s
    kB   = float(_em.kBoltzmann)      # J/K

    lam = _np.asarray(lam, dtype=_np.float64)
    T   = float(T)

    # invalids (avoid divide-by-zero / negatives)
    invalid = (T <= 0) | (lam <= 0)

    # helper: stable 1/(exp(x)-1)
    # - for small x: use 1/x - 1/2 + x/12  (from series)
    # - for large x: use exp(-x)
    # - otherwise:   1/expm1(x)
    def inv_expm1(x):
        x = _np.asarray(x, dtype=_np.float64)
        out = _np.empty_like(x)

        # thresholds tuned for float64
        small = 1e-6
        large = 50.0    # well below ~709 overflow threshold

        # small-x series
        xs = x < small
        if _np.any(xs):
            xs_val = x[xs]
            out[xs] = (1.0/xs_val) - 0.5 + xs_val/12.0

        # large-x: 1/(e^x - 1) ≈ e^{-x}
        xl = x > large
        if _np.any(xl):
            out[xl] = _np.exp(-x[xl])

        # mid range: safe to use expm1
        xm = ~(xs | xl)
        if _np.any(xm):
            out[xm] = 1.0/_np.expm1(x[xm])

        return out

    with _np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        if unit == 'wavelength':
            # meters
            ll = lam * 1e-6

            x = (h*c0) / (ll * kB * T)                # dimensionless
            denom = inv_expm1(x)                      # ≈ 1/(exp(x)-1), overflow-safe
            pref  = (2.0*h*c0*c0) / (ll**5)           # W·m^-3·sr^-1
            Ibb_m = pref * denom                      # W·m^-3·sr^-1
            Ibb   = Ibb_m * 1e-6                      # → W·m^-2·um^-1·sr^-1

        elif unit == 'frequency':
            # frequency from lam (lam passed in microns)
            ll = lam * 1e-6
            vv = c0 / ll                               # Hz

            x = (h*vv) / (kB*T)                        # dimensionless
            denom = inv_expm1(x)                       # ≈ 1/(exp(x)-1), overflow-safe
            pref  = (2.0*h * vv**3) / (c0*c0)          # W·m^-2·Hz^-1·sr^-1
            Ibb   = pref * denom
        else:
            raise ValueError("unit must be 'wavelength' or 'frequency'")

    # set invalids to nan (e.g., nonpositive T or lam)
    if _np.any(invalid):
        Ibb = _np.where(invalid, _np.nan, Ibb)

    return Ibb


def yCIE_lum(lam):
    '''
    CIE photoscopic luminosity function from Stockman & Sharpe as a function of wavelength

    Parameters
    ----------
    lam : 1D float array (or scalar)
        wavelength in um.

    Returns
    -------
    Interpolated CIE lum

    '''
    # interpolate values according to lam spectra
    lam = lam*1E3 # change units to nm
    
    yCIE = read_spectrafile(lam,'CIE_lum.txt', True)
    
    # keep only positive values
    yCIE = _np.clip(yCIE, 0, None)
    
    return yCIE

def spectral_average(lam_um: _np.ndarray,
                     spec_prop: _np.ndarray,
                     *,
                     spectrum: str = 'solar',
                     T: _Optional[float] = None,
                     bounds: _Optional[_Tuple[float, float]] = None) -> float:
    """
    Compute a spectrum-weighted mean of a spectral property over wavelength.

    Parameters
    ----------
    lam_um : np.ndarray
        Wavelengths in micrometers [um], shape (N,). Must be positive and finite.
    spec_prop : np.ndarray
        Spectral property sampled at `lam_um`, shape (N,). NaNs are allowed and will be masked.
    spectrum : str, optional
        Weighting spectrum selector:
          - 'solar'         : AM1.5 Global (W / m^2 / um)
          - 'solar:direct'  : AM1.5 Direct (W / m^2 / um)
          - 'thermal'       : Planck distribution at temperature `T` if provided,
                              else T = 300 K (W / m^2 / um / sr)
                              
    T : float or None, optional
        Black-body temperature in Kelvin. Only used when `spectrum` is 'thermal'.
        If None and `spectrum=='thermal'`, defaults to 300 K.
    bounds : (float, float) or None, optional
        Spectral interval [lam_min, lam_max] in micrometers to restrict the integration.
        If None, the full `lam_um` range is used.

    Returns
    -------
    float
        Spectrum-weighted mean:
            <prop> = ∫ prop(λ) w(λ) dλ / ∫ w(λ) dλ
        integrated with the trapezoidal rule over the (optionally bounded) input grid.

    Raises
    ------
    ValueError
        If inputs are malformed or not enough valid samples remain after masking/bounds.

    Notes
    -----
    - No resampling is performed; the input grid is used as-is.
    - Arrays are internally sorted by wavelength if needed.
    - Both numerator and denominator use the same validity mask.
    """
    lam_um = _np.asarray(lam_um, dtype=float)
    spec_prop   = _np.asarray(spec_prop,   dtype=float)

    # Basic checks
    if lam_um.ndim != 1 or spec_prop.ndim != 1:
        raise ValueError("`lam_um` and `spec_prop` must be 1D arrays.")
    if lam_um.size != spec_prop.size:
        raise ValueError("`lam_um` and `spec_prop` must have the same length.")
    if not _np.all(_np.isfinite(lam_um)) or _np.any(lam_um <= 0):
        raise ValueError("`lam_um` must be positive and finite.")

    # Sort grid if not strictly increasing
    if not _np.all(_np.diff(lam_um) > 0):
        order = _np.argsort(lam_um)
        lam_um = lam_um[order]
        spec_prop   = spec_prop[order]

    # Optional spectral bounds
    if bounds is not None:
        lam_min, lam_max = bounds
        if not _np.isfinite(lam_min) or not _np.isfinite(lam_max) or lam_min >= lam_max:
            raise ValueError("`bounds` must be (lam_min, lam_max) with lam_min < lam_max.")
        in_rng = (lam_um >= lam_min) & (lam_um <= lam_max)
        if not _np.any(in_rng):
            raise ValueError("No samples fall within `bounds`.")
        lam_um = lam_um[in_rng]
        spec_prop   = spec_prop[in_rng]

    s = spectrum.strip().lower()

    # Build weighting spectrum on the provided grid
    if s == 'solar':
        w = AM15(lam_um, spectra_type='global')
    elif s in ('solar:direct', 'solar-direct', 'direct'):
        w = AM15(lam_um, spectra_type='direct')
    elif s in ('thermal', 'planck', 'blackbody'):
        T_use = 300.0 if T is None else float(T)
        if not _np.isfinite(T_use) or T_use <= 0:
            raise ValueError("Temperature `T` must be a positive finite value.")
        w = Bplanck(lam_um, T=T_use, unit='wavelength')
    else:
        raise ValueError("`spectrum` must be 'solar', 'solar:direct', or 'thermal'.")

    # Mask invalids: NaN property, NaN/negative/zero weights
    mask = _np.isfinite(spec_prop) & _np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        raise ValueError("Not enough valid samples to integrate after masking/bounds.")

    lam = lam_um[mask]
    p   = spec_prop[mask]
    ww  = w[mask]

    # Weighted average over wavelength
    num = _np.trapz(p * ww, lam)
    den = _np.trapz(ww, lam)

    if den == 0 or not _np.isfinite(den):
        raise ValueError("Weight integral is zero or invalid; check chosen spectrum and bounds.")

    return float(num / den)

def plot_spectra(
        *curves: _Union[_Tuple[_Any, ...]],
        # Background toggles & params
        show_solar: bool = True,
        solar_spectrum: str = 'global',   # 'global' or 'direct'
        show_blackbody: bool = True,
        show_atmosphere: bool = True,
        atmosphere_tilt_deg: float = 0.0,
        Tbb: float = 300.0,
        Ebb_scale: float = 10.0,
        lam_background_samples: int = 2000,
        scale_atm_to_right_axis: bool = True,
        # Axes / styling
        ax: _Optional[_Any] = None, # axis to plot on (if None, create new figure)
        ax_twinx: _Optional[_Any] = None, # axis to twin plot (if None, create new y twin axis)
        xscale: str = 'log',              # 'log' or 'linear'
        xticks: _Optional[_Sequence[float]] = (0.3, 0.4, 0.75, 1.4, 3, 8, 15, 30, 50, 100),
        xlim: _Optional[_Tuple[float, float]] = None,
        ylim_left: _Optional[_Tuple[float, float]] = None,
        ylabel: _Optional[str] = None,
        right_ylabel: str = r"Spectral irradiance (W m$^{-2}$ μm$^{-1}$)",
        title: _Optional[str] = None,
        # Legend control
        curves_legend_loc: _Union[str, _Tuple[float, float]] = 'best',
        show_background_legend: bool = False,   # If True include background items in the legend
        curves_legend_kwargs: _Optional[_Dict[str, _Any]] = None,
        background_alpha: float = 0.22,
        figsize: _Tuple[float, float] = (10.0, 3.8),
        tight_layout: bool = True,
        # Behavior
        sort_wavelength: bool = True,
        validate_positive_x_for_log: bool = True,
    ):
    """
    Multi-curve spectral plotting (matplotlib forwarding style) with a SINGLE legend.

    Legend logic:
      - Curves (user-provided lines) are always listed first.
      - If show_background_legend=True, background entries (solar, blackbody, atmosphere)
        are appended after the curve entries in the SAME legend.
      - The legend placement is chosen using only the curve handles if curves_legend_loc='best'
        to avoid covering the user data. Then the background handles are added without
        recomputing placement (so placement is optimized for curves).

    Curve specification tuples (each forwarded to ax.plot):
        (x, y)
        (x, y, fmt)
        (x, y, fmt, {'label': 'R'})
        (x, y, {'label': 'R', 'color': 'red'})
        (x1, y1, x2, y2, fmt, {'label': 'multi'})
    If the last element is a dict it is treated as **kwargs; otherwise all are positional.

    Background labels:
        Solar irradiance:  AM1.5 (<type>)
        Blackbody:         $E_{\\mathrm{bb},\\lambda}(T) \\times s$
        Atmospheric trans.: $T_{\\mathrm{atm},\\lambda}$  (scaled if requested)

    Returns
    -------
    fig, ax
    """
    try:
        import matplotlib.pyplot as _plt
        from matplotlib.ticker import FuncFormatter as _FuncFormatter
    except ImportError as exc:
        raise ImportError("matplotlib is required for plot_multi_spectra_forward") from exc

    # Default legend kwargs (compact)
    if curves_legend_kwargs is None:
        curves_legend_kwargs = dict(handlelength=1.8, handletextpad=0.6, 
                                    borderaxespad=0.5, labelspacing=0.5)
    else:
        curves_legend_kwargs.setdefault('handlelength', 1.8)
        curves_legend_kwargs.setdefault('handletextpad', 0.6)
        curves_legend_kwargs.setdefault('borderaxespad', 0.5)
        curves_legend_kwargs.setdefault('labelspacing', 0.5)

    # --- Determine wavelength span from first elements of curve specs ---
    xs_for_span: _List[_np.ndarray] = []
    for spec in curves:
        if not isinstance(spec, (tuple, list)) or not spec:
            continue
        first = spec[0]
        if hasattr(first, "__len__") and not isinstance(first, (str, bytes, dict)):
            try:
                arr = _np.asarray(first, dtype=float)
                if arr.ndim == 1 and arr.size:
                    xs_for_span.append(arr)
            except Exception:
                pass
    if not xs_for_span:
        raise ValueError("No wavelength arrays detected in curve specifications.")
    lam_all = _np.concatenate(xs_for_span)
    lam_min = float(_np.nanmin(lam_all))
    lam_max = float(_np.nanmax(lam_all))
    if lam_min <= 0 and xscale == 'log':
        raise ValueError("Non-positive wavelengths encountered with xscale='log'.")

    lam_bg = (_np.logspace(_np.log10(lam_min), _np.log10(lam_max), lam_background_samples)
              if xscale == 'log'
              else _np.linspace(lam_min, lam_max, lam_background_samples))

    # --- Figure and axis setup ---
    # If no axis provided, create a new figure/axes.  Otherwise reuse the supplied
    # axis and get its figure.  For external axes we do not override the figure size.
    created_fig = False
    if ax is None:
        fig, ax = _plt.subplots(figsize=figsize)
        created_fig = True
    else:
        if not isinstance(ax, _plt.Axes):
            raise ValueError("`ax` must be a matplotlib Axes instance or None.")
        fig = ax.figure
        ax.patch.set_alpha(0.0) # transparent primary axis background 

    # Create a twin axis for the backgrounds.  Lower its z-order and hide its patch
    # so that the filled backgrounds appear behind the user’s curves:contentReference[oaicite:2]{index=2}.
    if ax_twinx is None:
        ax2 = ax.twinx()
    else:
        if not isinstance(ax_twinx, _plt.Axes):
            raise ValueError("`ax_twinx` must be a matplotlib Axes instance or None.")
        ax2 = ax_twinx
    ax2.patch.set_alpha(0.0)

    curve_handles: _List[_Any] = []
    background_handles: _List[_Any] = []
    right_axis_max = 0.0

    # --- Solar background ---
    if show_solar:
        spec_type = 'global' if solar_spectrum.lower().startswith('g') else 'direct'
        Gsun = AM15(lam_bg, spectra_type=spec_type)
        right_axis_max = max(right_axis_max, float(_np.nanmax(Gsun)))
        h_sun = ax2.fill_between(
            lam_bg, Gsun,
            color="#f6f600", alpha=background_alpha, lw=0,
            label=f"AM1.5 ({spec_type})"
        )
        background_handles.append(h_sun)
    else:
        spec_type = 'global'

    # --- Blackbody background ---
    if show_blackbody:
        Ebb = _np.pi * Bplanck(lam_bg, Tbb, unit='wavelength')  # W/m^2/μm
        Ebb_scaled = Ebb * Ebb_scale
        right_axis_max = max(right_axis_max, float(_np.nanmax(Ebb_scaled)))
        h_bb = ax2.fill_between(
            lam_bg, Ebb_scaled,
            color="#e10000", alpha=background_alpha, lw=0,
            label=r"$E_{\mathrm{bb},\lambda}(%d\,\mathrm{K}) \times %g$" % (int(Tbb), Ebb_scale)
        )
        background_handles.append(h_bb)

    if right_axis_max <= 0:
        right_axis_max = 1.0

    # --- Atmospheric transmittance background ---
    atm_scale_factor = 1.0
    if show_atmosphere:
        Tatm = T_atmosphere_hemi(lam_bg, beta_tilt=atmosphere_tilt_deg)
        if scale_atm_to_right_axis:
            atm_scale_factor = right_axis_max
        Tatm_scaled = Tatm * atm_scale_factor
        h_atm = ax2.fill_between(
            lam_bg, Tatm_scaled,
            color="#0013e1", alpha=background_alpha, lw=0,
            label=r"$T_{\mathrm{atm},\lambda}$"
        )
        background_handles.append(h_atm)

    # --- User curve plotting (forward args) ---
    for spec in curves:
        if not isinstance(spec, (tuple, list)) or not spec:
            continue
        if isinstance(spec[-1], dict):
            *positional, kw = spec
            kwargs = kw.copy()
        else:
            positional = list(spec)
            kwargs = {}

        # Optional sorting for simple (x,y[,fmt]) pattern
        if sort_wavelength and len(positional) >= 2:
            try:
                x_arr = _np.asarray(positional[0], dtype=float)
                y_arr = _np.asarray(positional[1], dtype=float)
                if x_arr.ndim == 1 and y_arr.ndim == 1 and x_arr.shape == y_arr.shape:
                    if xscale == 'log' and validate_positive_x_for_log and _np.any(x_arr <= 0):
                        raise ValueError("Non-positive wavelength values in a curve with xscale='log'.")
                    if not _np.all(_np.diff(x_arr) > 0):
                        order = _np.argsort(x_arr)
                        positional[0] = x_arr[order]
                        positional[1] = y_arr[order]
            except Exception:
                pass  # Let matplotlib raise if truly problematic

        line_list = ax.plot(*positional, **kwargs)
        if line_list:
            curve_handles.append(line_list[0])

    # --- Axis formatting ---
    ax.set_xscale(xscale)
    if xticks is not None:
        ax.set_xticks(xticks)
        from matplotlib.ticker import FuncFormatter as _FuncFormatter
        ax.xaxis.set_major_formatter(_FuncFormatter(lambda v, _pos: f"{v:g}"))

    ax.set_xlim(*(xlim if xlim is not None else (lam_min, lam_max)))
    if ylim_left is not None:
        ax.set_ylim(*ylim_left)

    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax2.set_ylabel(right_ylabel)
    if title: ax.set_title(title)
    if ylabel: ax.set_ylabel(ylabel)

    # --- Legend assembly ---
    if curve_handles:
        # Determine placement using only curves if 'best'
        if curves_legend_loc == 'best':
            temp_leg = ax.legend(handles=curve_handles,
                                 frameon=False,
                                 loc='best',
                                 **curves_legend_kwargs)
            # Capture chosen location (internal attribute _loc may hold code or str)
            chosen_loc = temp_leg._loc
            temp_leg.remove()
        else:
            chosen_loc = curves_legend_loc

        if show_background_legend and background_handles:
            combined_handles = curve_handles + background_handles
        else:
            combined_handles = curve_handles

        ax.legend(handles=combined_handles,
                  frameon=False,
                  loc=chosen_loc,
                  **curves_legend_kwargs)
    else:
        # If no curve handles but backgrounds requested, still show background legend
        if show_background_legend and background_handles:
            ax.legend(handles=background_handles,
                      frameon=False,
                      loc=curves_legend_loc,
                      **curves_legend_kwargs)

    # Only apply tight_layout when we created a new figure to avoid disturbing an
    # externally managed figure layout:contentReference[oaicite:3]{index=3}.
    if tight_layout and created_fig:
        fig.tight_layout()

    return fig, ax