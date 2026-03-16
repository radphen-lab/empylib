# -*- coding: utf-8 -*-
"""
Library of radiative transfer function

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import sys

# empylib_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.insert(0,empylib_folder)

import numpy as _np
from . import miescattering as mie
from . import waveoptics as wv
from . import nklib as nk
import iadpython as _iad
import pandas as pd
from typing import Union as _Union, Optional as _Optional, List as _List, Tuple as _Tuple
from .utils import _as_carray, _check_mie_inputs, _hide_signature
from .nklib import emt_brugg, emt_multilayer_sphere
from inspect import Signature, signature as _signature

_IAD_SUPPORTS_TABULATED_PF = "pf_type" in _signature(_iad.Sample.__init__).parameters

@_hide_signature
def T_beer_lambert(lam: _Union[float, _np.ndarray],                                         # wavelengths [µm]
                   Nh: _Union[float, _np.ndarray],                                     # host refractive index
                   Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],  # particle refractive index
                   D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],   # sphere diameters [µm] 
                   fv: float,                                                          # film volume fraction 
                   tfilm: float, 
                   *,
                   theta: _Union[float, _np.ndarray]= 0.0,                            # angle of incidence in degrees
                   Nup: _Union[float, _np.ndarray] = 1.0,                              # refractive index above
                   Ndw: _Union[float, _np.ndarray] = 1.0,                              # refractive index below
                   size_dist: _np.ndarray = None,                                      # number-fraction weights p_i 
                   dependent_scatt = False,                                            # use Perkus-Yevik for dependent scattering
                   effective_medium: bool = False,                                     # whether to compute effective Nh via Bruggeman
                   use_phase_fun: bool = False,                                        # whether to use phase function instead of g
                   check_inputs = True                                                 # whether to check mie inputs
                   ):
    '''
    Transmittance and reflectance from Beer-Lamberts law for a film with 
    spherical particles. Reflectance is computed from classical formulas for
    incoherent light incident on a slab between two semi-infinite media 
    (no scattering is considered for this parameter)

    Parameters
    lam : array-like, shape (nλ,)
        Wavelengths [µm], strictly positive.

    Nh : float or array-like (nλ,)
        Host refractive index (can be complex). If array-like, length must equal len(lam).

    Np (float, 1darray or list): Complex refractive index of each 
                                            shell layer. Np.shape[1] == len(D). 
        Options are:
        float:   solid sphere and constant refractive index
        1darray: solid sphere and spectral refractive index (len = lam)
        list:    multilayered sphere (with both constant or spectral refractive indexes)
    
    D : float, _np.ndarray or list
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        if multilayer sphere, use list of floats (monodisperse) or arrays (polydisperse).
    
    fv : float
        Particle volume fraction in (0, 1). Used only to compute an effective medium Nh via
        `nk.emt_brugg(fv, Np, Nh)`.

    tfilm : float
        Film thickness [mm], ≥ 0.

    theta : float or array-like, optional
        Angle of incidence in degrees. Default is 0 (normal incidence). If array-like, length must equal len(lam).

    Nup : float or array-like, optional
        Refractive index above the film. Default is 1.0 (air). If array-like, length must equal len(lam).

    Ndw : float or array-like, optional
        Refractive index below the film. Default is 1.0 (air). If array-like, length must equal len(lam).

    size_dist : array-like, shape (n_bins,), optional
        Number-fraction weights for polydisperse particles. Default is None (monodisper
        particles). If given, must be 1D array with len(size_dist) == len(D[0]) (if D is list)
        or len(size_dist) == len(D) (if D is array).
        The weights must be nonnegative and sum to 1.
        The size distribution is used only when `dependent_scatt=True`.
    
    dependent_scatt : bool, optional
        Whether to include dependent scattering effects via Percus-Yevick structure factor
        (default: False; not recommended for metallic spheres or high fv)
    
    effective_medium : bool, optional
        Whether to compute an effective medium for the host refractive index via Bruggeman
        (default: False; recommended for fv >~ 0.1)
    
    use_phase_fun : bool, optional
        Whether to use the full phase function in the radiative transfer (default: False).
        If False, the asymmetry parameter g is used instead (Henyey-Greenstein approximation).
        Using the phase function is more accurate but also more computationally intensive.
    
    check_inputs : bool, optional
        Whether to check mie inputs (default: True)    

    Returns
    - results_df : pd.DataFrame with index=lam and columns:
                'Rtot' : total reflectance
                'Ttot' : total transmittance
                'Tspec': specular (unscattered) transmittance
                'Tdif': diffuse (scattered) transmittance
    '''

    # ---------- coerce arrays & basic checks ----------
    if check_inputs:
        lam, Nh, Np, D, size_dist = _check_mie_inputs(lam, Nh, Np, D, 
                                                      size_dist=size_dist)

    nlam = lam.size
    Nup = _as_carray(Nup, "Nup", nlam, val_type=complex)
    Ndw = _as_carray(Ndw, "Ndw", nlam, val_type=complex)

    if not (0 <= float(fv) < 1):
        raise ValueError("fv (volume fraction) must be in [0,1).")
    if not _np.isscalar(tfilm) or tfilm < 0:
        raise ValueError("tfilm must be a nonnegative scalar in mm.")

    # if dependent scatt, check that particle is metallic through: Im(Np) > Re(Np)
    if dependent_scatt:
        if _np.any(_np.array(Np).real < _np.array(Np).imag):
            print("Warning: Dependent scattering theory not recommended for metallic particles.")
    
    # ---------- Effective medium for host (if your convention is to dress Nh) ----------
    N_layers = len(D)                                    # number of layers in the sphere
    if effective_medium:
        # Compute mean diameter of each layer
        D_layers_mean = []
        for i in range(N_layers):
            if size_dist is None:
                # Monodisperse
                D_layers_mean.append(float(_np.asarray(D[i]).ravel()[0]))
            else:
                # Polydisperse
                D_layers_mean.append(_np.average(D[i], axis=0,   # -> float
                                            weights=size_dist))  # size_dist shape (n_bins,)

        # Compute effective refractive index of host using Bruggeman EMT                                   
        Np_eff = emt_multilayer_sphere(D_layers_mean, Np, check_inputs=False)
        Nh = emt_brugg(fv, Np_eff, Nh)

    # ---------- Mie cross sections and phase function ----------
    cabs, csca, _, _ = mie.cross_section_ensemble(lam, Nh, Np, D, fv, 
                                                  size_dist=size_dist,
                                                  check_inputs=False,
                                                  effective_medium=False,
                                                  dependent_scatt=dependent_scatt,
                                                  phase_function=use_phase_fun)

    # ---------- n_tot and coefficients (µm⁻¹) ----------
    # Particle volume (or mean volume if polydisperse)
    V  = (4.0 / 3.0) * _np.pi * (D[-1] / 2.0) ** 3  # [µm³]
    if size_dist is not None:
            V = float(_np.sum(size_dist * V))              # ⟨V⟩ [µm³]

    # Get scattering and absorption coefficients
    n_tot = fv / V                # [µm⁻³]
    k_sca = n_tot * csca          # [µm⁻¹]
    k_abs = n_tot * cabs          # [µm⁻¹]
    k_ext = k_sca + k_abs         # [µm⁻¹]

    # ---------- Fresnel reflectance/transmittance ----------
    tfilm = tfilm*1E3 # convert mm to micron units

    theta_rad = _np.radians(theta)
    Rp, Tp = wv.incoh_multilayer(
        lam,
        N_layers=[Nh],
        thickness=tfilm,
        aoi=theta_rad,
        N_above=Nup,
        N_below=Ndw,
        polarization='TM',
    )
    Rs, Ts = wv.incoh_multilayer(
        lam,
        N_layers=[Nh],
        thickness=tfilm,
        aoi=theta_rad,
        N_above=Nup,
        N_below=Ndw,
        polarization='TE',
    )
    T    = (Ts + Tp)/2
    Rtot = (Rp + Rs)/2
    
    theta1 = wv.snell(Nup, Nh, theta_rad)
        
    Ttot = T*_np.exp(-k_abs*tfilm/_np.cos(theta1.real))
    Tspec = T*_np.exp(-k_ext*tfilm/_np.cos(theta1.real))
    Tdif = Ttot - Tspec

    # store data into a dataframe (λ index)
    results_df = pd.DataFrame({
        'Rtot': Rtot,
        'Ttot': Ttot,
        'Tspec': Tspec,
        'Tdif': Tdif
    }, index=lam)

    results_df.index.name = 'Wavelength (µm)'

    return results_df

@_hide_signature
def adm_sphere(lam: _Union[float, _np.ndarray],                                     # wavelengths [µm]
                Nh: _Union[float, _np.ndarray],                                     # host refractive index
                Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],  # particle refractive index
                D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],   # sphere diameters [µm] 
                fv: float,                                                          # film volume fraction 
                tfilm: float,                                                       # film thickness [mm]       
                *,
                Nup: _Union[float, _np.ndarray] = 1.0,                              # refractive index above
                Ndw: _Union[float, _np.ndarray] = 1.0,                              # refractive index below
                size_dist: _np.ndarray = None,                                      # number-fraction weights p_i 
                dependent_scatt = False,                                            # use Perkus-Yevik for dependent scattering
                effective_medium: bool = False,                                     # whether to compute effective Nh via Bruggeman
                use_phase_fun: bool = False,                                        # whether to use phase function instead of g
                cone_incidence: _Optional[_Tuple[float, float]] = None,             # (theta_min, theta_max) in degrees for diffuse cone incidence
                lambertian: bool = False                                            # whether to compute Lambertian incidence instead of normal
                ):
    '''
    Parameters
    lam : array-like, shape (nλ,)
        Wavelengths [µm], strictly positive.

    Nh : float or array-like (nλ,)
        Host refractive index (can be complex). If array-like, length must equal len(lam).

    Np (float, 1darray or list): Complex refractive index of each 
                                            shell layer. Np.shape[1] == len(D). 
        Options are:
        float:   solid sphere and constant refractive index
        1darray: solid sphere and spectral refractive index (len = lam)
        list:    multilayered sphere (with both constant or spectral refractive indexes)
    
    D : float, _np.ndarray or list
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        if multilayer sphere, use list of floats (monodisperse) or arrays (polydisperse).
    
    fv : float
        Particle volume fraction in (0, 1). Used only to compute an effective medium Nh via
        `nk.emt_brugg(fv, Np, Nh)`.

    tfilm : float
        Film thickness [mm], ≥ 0.
            
    Nup : float or array-like, optional
        Refractive index above the film. Default is 1.0 (air). If array-like, length must equal len(lam).

    Ndw : float or array-like, optional
        Refractive index below the film. Default is 1.0 (air). If array-like, length must equal len(lam).

    size_dist : array-like, shape (n_bins,), optional
        Number-fraction weights for polydisperse particles. Default is None (monodisper
        particles). If given, must be 1D array with len(size_dist) == len(D[0]) (if D is list)
        or len(size_dist) == len(D) (if D is array).
        The weights must be nonnegative and sum to 1.
    
    dependent_scatt : bool, optional
        Whether to include dependent scattering effects via Percus-Yevick structure factor
        (default: False; not recommended for metallic spheres or high fv)
    
    effective_medium : bool, optional
        Whether to compute an effective medium for the host refractive index via Bruggeman
        (default: False; recommended for fv >~ 0.1)
    
    use_phase_fun : bool, optional
        Whether to use the full phase function in the radiative transfer (default: False).
        If False, the asymmetry parameter g is used instead (Henyey-Greenstein approximation).
        Using the phase function is more accurate but also more computationally intensive.

    cone_incidence : tuple, optional
        Optional tuple (theta_min, theta_max) in degrees for diffuse cone incidence    
    
    lambertian : bool, optional
        Whether to compute Lambertian incidence instead of normal incidence

    Returns
    - results_df : pd.DataFrame with index=lam and columns:
                'Rtot' : total reflectance
                'Ttot' : total transmittance
                'Rspec': specular (unscattered) reflectance
                'Tspec': specular (unscattered) transmittance
                'Rdif' : diffuse reflectance
                'Tdif' : diffuse transmittance
                'Rcone': reflectance for cone diffuse incidence (if requested)
                'Tcone': transmittance for cone diffuse incidence (if requested)
                'Rlam' : reflectance for Lambertian incidence (if requested)
                'Tlam' : transmittance for Lambertian incidence (if requested)
    '''
    # ---------- coerce arrays & basic checks ----------
    lam, Nh, Np, D, size_dist = _check_mie_inputs(lam, Nh, Np, D, size_dist=size_dist)

    nlam = lam.size
    Nup = _as_carray(Nup, "Nup", nlam, val_type=complex)
    Ndw = _as_carray(Ndw, "Ndw", nlam, val_type=complex)

    if not (0 <= float(fv) < 1):
        raise ValueError("fv (volume fraction) must be in [0,1).")
    if not _np.isscalar(tfilm) or tfilm < 0:
        raise ValueError("tfilm must be a nonnegative scalar in mm.")

    # if dependent scatt, check that particle is metallic through: Im(Np) > Re(Np)
    # if dependent_scatt:
    #     if _np.any(_np.array(Np).real < _np.array(Np).imag):
    #         print("Warning: Dependent scattering theory not recommended for metallic particles.")

    # ---------- Effective medium for host (if your convention is to dress Nh) ----------
    N_layers = len(D)                                    # number of layers in the sphere
    
    Nh_eff = Nh.copy()
    if effective_medium and fv > 0.0:
        # Compute mean diameter of each layer
        D_layers_mean = []
        for i in range(N_layers):
            if size_dist is None:
                # Monodisperse
                D_layers_mean.append(float(_np.asarray(D[i]).ravel()[0]))
            else:
                # Polydisperse
                D_layers_mean.append(_np.average(D[i], axis=0,   # -> float
                                            weights=size_dist))  # size_dist shape (n_bins,)

        # Compute effective refractive index of host using Bruggeman EMT                                   
        Np_eff = emt_multilayer_sphere(D_layers_mean, Np, check_inputs=False)
        Nh_eff = emt_brugg(fv, Np_eff, Nh)
    
    # ---------- Mie cross sections and phase function ----------
    theta_eval = _np.linspace(0, _np.pi, 100)
    cabs, csca, gcos, phase_scatter = mie.cross_section_ensemble(lam, Nh_eff, Np, D, fv, 
                                                                size_dist=size_dist,
                                                                theta=theta_eval,
                                                                check_inputs=False,
                                                                effective_medium=False,
                                                                dependent_scatt=dependent_scatt,
                                                                phase_function=use_phase_fun)

    # ---------- n_tot and coefficients (µm⁻¹) ----------
    # Particle volume (or mean volume if polydisperse)
    V  = (4.0 / 3.0) * _np.pi * (D[-1] / 2.0) ** 3  # [µm³]
    if size_dist is not None:
            V = float(_np.sum(size_dist * V))              # ⟨V⟩ [µm³]

    # Get scattering and absorption coefficients
    n_tot = fv / V                # [µm⁻³]
    k_sca = n_tot * csca          # [µm⁻¹]
    k_abs = n_tot * cabs          # [µm⁻¹]

    # ---------- radiative transfer ----------
    if use_phase_fun:
        df_results = adm(lam, tfilm, k_sca, k_abs, Nh=Nh_eff, 
                                       phase_fun=phase_scatter, 
                                       Nup=Nup, 
                                       Ndw=Ndw,
                                       cone_incidence=cone_incidence,
                                       lambertian=lambertian)
    else:
        df_results = adm(lam, tfilm, k_sca, k_abs, Nh=Nh_eff, 
                                       gcos=gcos, 
                                       Nup=Nup, 
                                       Ndw=Ndw,
                                       cone_incidence=cone_incidence,
                                       lambertian=lambertian)

    return df_results

@_hide_signature
def adm(lam, tfilm, k_sca, k_abs, Nh,
        gcos=None,                                              # optional: asymmetry parameter per λ
        *,
        phase_fun=None,                                         # optional: phase function vs θ (DataFrame only; θ index in degrees 0..180)
        Nup=1.0,                                                # refractive index above
        Ndw=1.0,                                                # refractive index below
        quad_pts: int = 16,                                     # IAD quadrature points when using a tabulated PF
        cone_incidence: _Optional[_Tuple[float, float]] = None, # (theta_min, theta_max) in degrees for diffuse cone incidence
        lambertian: bool = False                                # whether to compute Lambertian incidence instead of normal
):
    """
    Adding–doubling (IAD) reflectance/transmittance for a scattering/absorbing film.

    Inputs (arrays are per-wavelength unless noted):
    - lam      : (nλ,) wavelengths [µm]
    - tfilm    : scalar film thickness [mm]
    - k_sca    : (nλ,) scattering coefficient [µm^-1]
    - k_abs    : (nλ,) absorption coefficient [µm^-1]
    - Nh       : scalar or (nλ,) complex refractive index of the film host
    --------------------------------------------------------------------------
    Choose ONE angular description:
    - gcos     : (nλ,) asymmetry parameter  (Henyey–Greenstein style)
      OR
    - phase_fun: pd.DataFrame of the differential *phase function* (not normalized to 1/4π),
                 shape (nθ, nλ). **Index must be θ in degrees from 0 to 180.**
                 Columns must be the wavelengths (same values as `lam`, order-agnostic).
                 The function will convert θ→μ=cosθ and sort μ ascending in [-1, 1].
    --------------------------------------------------------------------------
    - Nup, Ndw : scalar or (nλ,) complex refractive indices above/below (defaults=1.0)
    - quad_pts : quadrature points for IAD when using a tabulated phase function
    - cone_incidence : optional tuple (theta_min, theta_max) in degrees for
                       diffuse cone incidence calculations

    Returns:
    - results_df : pd.DataFrame with index=lam and columns:
                'Rtot' : total reflectance
                'Ttot' : total transmittance
                'Rspec': specular (unscattered) reflectance
                'Tspec': specular (unscattered) transmittance
                'Rdif' : diffuse reflectance
                'Tdif' : diffuse transmittance
                'Rcone': reflectance for cone diffuse incidence (if requested)
                'Tcone': transmittance for cone diffuse incidence (if requested)
                'Rlam' : reflectance for Lambertian incidence (if requested)
                'Tlam' : transmittance for Lambertian incidence (if requested)
    """
    # ---------- coerce arrays ----------
    lam   = _np.atleast_1d(_np.asarray(lam,   float))
    k_sca = _np.atleast_1d(_np.asarray(k_sca, float))
    k_abs = _np.atleast_1d(_np.asarray(k_abs, float))

    if lam.ndim != 1:
        raise ValueError("lam must be a 1D array of wavelengths [µm].")
    nlam = lam.size
    for name, arr in [("k_sca", k_sca), ("k_abs", k_abs)]:
        if _np.asarray(arr).shape != (nlam,):
            raise ValueError(f"{name} must have the same length as lam.")

    Nh_arr  = _as_carray(Nh,  "Nh" , nlam, val_type=complex)
    Nup_arr = _as_carray(Nup, "Nup", nlam, val_type=complex)
    Ndw_arr = _as_carray(Ndw, "Ndw", nlam, val_type=complex)

    # keep all positive
    k_sca = _np.maximum(k_sca, 0.0)
    k_abs = _np.maximum(k_abs, 0.0)
    Nh_arr.imag = _np.maximum(Nh_arr.imag, 0.0)

    # ---------- convert to IAD units (mm^-1); include host absorption via Im(n) ----------
    # k_sca, k_abs are in µm^-1 -> mm^-1 multiply by 1e3
    mu_s = k_sca * 1e3

    # host material absorption: α_host = 4π k / λ  (λ in µm)  -> in mm^-1 multiply by 1e3
    kz_imag = 2.0 * _np.pi / lam * Nh_arr.imag * 1e3  # (2π/λ)*k in mm^-1
    mu_a = k_abs * 1e3 + 2.0 * kz_imag                # = (k_abs + 2*(2π/λ)k)*1e3 = (k_abs + 4πk/λ)*1e3
    mu_t = mu_s + mu_a
    d = float(tfilm)

    # ---------- choose angular description ----------
    use_pf = phase_fun is not None
    if use_pf and (gcos is not None):
        raise ValueError("Provide either gcos OR phase_fun, not both.")
    if (not use_pf) and (gcos is None):
        raise ValueError("You must provide gcos (per λ) or a tabulated phase_fun.")

    if not use_pf:
        gcos = _np.atleast_1d(_np.asarray(gcos, float))
        if gcos.shape != (nlam,):
            raise ValueError("gcos must have shape (len(lam),).")

    # ---------- prepare phase function (TABULATED path) ----------
    if use_pf:
        if not isinstance(phase_fun, pd.DataFrame):
            raise TypeError("phase_fun must be a pandas DataFrame with θ-degree index in [0,180].")

        # Validate θ index: must be degrees from 0 to 180
        theta_idx = _np.asarray(phase_fun.index, float)
        if theta_idx.ndim != 1:
            raise ValueError("phase_fun index must be 1D θ (degrees).")
        if theta_idx.min() < 0.0 or theta_idx.max() > 180.0:
            raise ValueError("phase_fun index (θ) must lie within [0°, 180°].")

        # Align columns to lam (order-agnostic but values must match)
        try:
            # If columns are numeric wavelengths, reindex to lam exactly (no interpolation here)
            PF = phase_fun.reindex(columns=lam, copy=False).values
        except Exception as e:
            raise ValueError("phase_fun columns must match lam (wavelengths).") from e
        if PF.shape != (theta_idx.size, nlam):
            raise ValueError("phase_fun must have shape (nθ, nλ) matching lam.")

        if _IAD_SUPPORTS_TABULATED_PF:
            # Convert θ → μ and sort ascending in [-1, 1]
            mu = _np.cos(_np.radians(theta_idx))
            order = _np.argsort(mu)    # ascending
            mu = mu[order]
            PF = PF[order, :]

            # IAD expects a DataFrame with index μ in [-1,1], one column per λ
            pf_df = pd.DataFrame(PF, index=mu, columns=lam)
            pf_df.index.name = "cos(theta)"
        else:
            # Older iadpython releases only expose Henyey-Greenstein anisotropy.
            # Collapse the supplied phase function to an equivalent g(λ) so the
            # public `phase_fun=` entrypoint remains usable in those environments.
            _, gcos = mie.scatter_from_phase_function(phase_fun)
            use_pf = False

    # ---------- run IAD per wavelength ----------
    Ttot  = _np.zeros(nlam, float)
    Rtot  = _np.zeros(nlam, float)
    Tspec = _np.zeros(nlam, float)
    Rspec = _np.zeros(nlam, float)

    if cone_incidence is not None:
        Tcone = _np.zeros(nlam, float)
        Rcone = _np.zeros(nlam, float)

    if lambertian:
        Rlam = _np.zeros(nlam, float)
        Tlam = _np.zeros(nlam, float)

    for j in range(nlam):
        # guard: IAD wants n >= 1
        n_real = float(max(Nh_arr.real[j], 1.0))
        n_up   = float(max(Nup_arr.real[j], 1.0))
        n_dw   = float(max(Ndw_arr.real[j], 1.0))

        if mu_t[j] <= 0.0:
            # transparent, non-scattering layer -> Fresnel only
            s = _iad.Sample(a=0.0, b=0.0, g=0.0, d=d, n=n_real, n_above=n_up, n_below=n_dw)
        else:
            a = mu_s[j] / mu_t[j]     # single-scattering albedo
            b = mu_t[j] * d           # optical thickness
            if not use_pf:
                s = _iad.Sample(a=a, b=b, g=float(gcos[j]), d=d,
                                n=n_real, n_above=n_up, n_below=n_dw)
            else:
                # tabulated phase function path
                pf_col = pf_df.iloc[:, j].to_frame()
                s = _iad.Sample(a=a, b=b, d=d, n=n_real, n_above=n_up, n_below=n_dw,
                                quad_pts=int(quad_pts),
                                pf_type="TABULATED", pf_data=pf_col)

        # total RT at normal and Lambertian (diffuse) incidence
        R, _, T, _ = s.rt_matrices()
        R_tot, T_tot, R_lam, T_lam = s.UX1_and_UXU(R, T)
        
        # specular RT (at normal incidence)
        R_spec, T_spec = s.unscattered_rt()

        Ttot[j]  = float(T_tot)
        Rtot[j]  = float(R_tot)
        Tspec[j] = float(T_spec)
        Rspec[j] = float(R_spec)
        if lambertian:
            Rlam[j] = float(R_lam)
            Tlam[j] = float(T_lam)

        # RT components for cone diffuse incidence (if requested)
        if cone_incidence is not None:
            # extract cone incident angles
            theta_min, theta_max = cone_incidence

            # get equivalent nu = cos(theta)
            nu_min = _np.cos(_np.radians(theta_max))  
            nu_max = _np.cos(_np.radians(theta_min))

            # compute diffuse cone incidence
            R_cone, T_cone = s.rt_diffuse_cone(R, T, nu_min, nu_max)
            Rcone[j] = float(R_cone)
            Tcone[j] = float(T_cone)

    # Compute diffuse components of R and T
    Rdif = Rtot - Rspec
    Tdif = Ttot - Tspec

    # Store data into a dataframe (λ index)
    results_df = pd.DataFrame({
        "Rtot": Rtot,
        "Ttot": Ttot,
        "Rspec": Rspec,
        "Tspec": Tspec,
        "Rdif": Rdif,
        "Tdif": Tdif
    }, index=lam)

    results_df.index.name = 'Wavelength (µm)'

    if cone_incidence is not None:
        results_df["Rcone"] = Rcone
        results_df["Tcone"] = Tcone
    
    if lambertian:
        results_df["Rlam"] = Rlam
        results_df["Tlam"] = Tlam

    return results_df
