# -*- coding: utf-8 -*-
"""
Library of radiative transfer function

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import numpy as _np
from . import miescattering as _mie
from . import waveoptics as _wv
import iadpython as _iad
import pandas as _pd
from typing import Union as _Union, Optional as _Optional, List as _List, Tuple as _Tuple
from .utils import (
    _as_1d_array,
    _check_mie_inputs,
    _hide_signature,
    _effective_host,
    _estimate_theta_npts,
    _make_angular_grid,
    _prepare_tabulated_phase_fun_for_iad,
)
from .nklib import emt_brugg as _emt_brugg, emt_multilayer_sphere as _emt_multilayer_sphere
from inspect import signature as _signature

__all__ = ('T_beer_lambert', 'adm_sphere', 'adm')

_IAD_SUPPORTS_TABULATED_PF = "pf_type" in _signature(_iad.Sample.__init__).parameters

@_hide_signature
def T_beer_lambert(wavelength: _Union[float, _np.ndarray],                             # wavelengths [µm]
                   N_host: _Union[float, _np.ndarray],                                 # host refractive index
                   N_particle: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],  # particle refractive index
                   D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],   # sphere diameters [µm] 
                   fv: float,                                                                  # film volume fraction 
                   thickness: float, 
                   *,
                   aoi: _Union[float, _np.ndarray]= 0.0,                               # angle of incidence (radians)
                   N_above: _Union[float, _np.ndarray] = 1.0,                          # refractive index above
                   N_below: _Union[float, _np.ndarray] = 1.0,                          # refractive index below
                   size_dist: _np.ndarray = None,                                      # number-fraction weights p_i 
                   dependent_scatt = False,                                            # use Perkus-Yevik for dependent scattering
                   effective_medium: bool = False,                                     # whether to compute effective N_host via Bruggeman
                   use_phase_fun: bool = False,                                        # whether to use phase function instead of g
                   check_inputs = True                                                 # whether to check mie inputs
                   ):
    '''
    Transmittance and reflectance from Beer-Lamberts law for a film with 
    spherical particles. Reflectance is computed from classical formulas for
    incoherent light incident on a slab between two semi-infinite media 
    (no scattering is considered for this parameter)

    Parameters
    wavelength : array-like, shape (nλ,)
        Wavelengths [µm], strictly positive.

    N_host : float or array-like (nλ,)
        Host refractive index (can be complex). If array-like, length must equal len(wavelength).

    N_particle (float, 1darray or list): Complex refractive index of each 
                                            shell layer. N_particle.shape[1] == len(D). 
        Options are:
        float:   solid sphere and constant refractive index
        1darray: solid sphere and spectral refractive index (len = wavelength)
        list:    multilayered sphere (with both constant or spectral refractive indexes)
    
    D : float, _np.ndarray or list
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        if multilayer sphere, use list of floats (monodisperse) or arrays (polydisperse).
    
    fv : float
        Particle volume fraction in (0, 1). Used only to compute an effective medium N_host via
        `nk.emt_brugg(fv, N_particle, N_host)`.

    thickness : float
        Film thickness [mm], ≥ 0.

    aoi : float or array-like, optional
        Angle of incidence in radians. Default is 0 (normal incidence). If array-like, length must equal len(wavelength).

    N_above : float or array-like, optional
        Refractive index above the film. Default is 1.0 (air). If array-like, length must equal len(wavelength).

    N_below : float or array-like, optional
        Refractive index below the film. Default is 1.0 (air). If array-like, length must equal len(wavelength).

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
    - results_df : _pd.DataFrame with index=wavelength and columns:
                'Rtot' : total reflectance
                'Ttot' : total transmittance
                'Tspec': specular (unscattered) transmittance
                'Tdif': diffuse (scattered) transmittance
    '''

    # ---------- coerce arrays & basic checks ----------
    if check_inputs:
        wavelength, N_host, N_particle, D, size_dist = _check_mie_inputs(wavelength, N_host, N_particle, D, 
                                                      size_dist=size_dist)

    n_wavelengths = wavelength.size
    N_above = _as_1d_array(N_above, "N_above", n_wavelengths=n_wavelengths, dtype=complex)
    N_below = _as_1d_array(N_below, "N_below", n_wavelengths=n_wavelengths, dtype=complex)

    if aoi > _np.pi/2:
        raise ValueError("aoi > pi/2")

    if not (0 <= float(fv) < 1):
        raise ValueError("fv (volume fraction) must be in [0,1).")
    if not _np.isscalar(thickness) or thickness < 0:
        raise ValueError("thickness must be a nonnegative scalar in mm.")

    # if dependent scatt, check that particle is metallic through: Im(N_particle) > Re(N_particle)
    if dependent_scatt:
        if _np.any(_np.array(N_particle).real < _np.array(N_particle).imag):
            print("Warning: Dependent scattering theory not recommended for metallic particles.")
    
    # ---------- Effective medium for host (if your convention is to dress N_host) ----------
    N_host_eff = N_host.copy()
    if effective_medium and fv > 0.0:
        N_host_eff = _effective_host(fv, N_particle, N_host, D, size_dist,
                                    emt_multilayer_fn=_emt_multilayer_sphere,
                                    emt_brugg_fn=_emt_brugg,
                                    )

    # ---------- Mie cross sections and phase function ----------
    cabs, csca, _, _ = _mie.cross_section_ensemble(wavelength, N_host_eff, N_particle, D, fv, 
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
    thickness = thickness*1E3 # convert mm to micron units

    theta_rad = aoi
    Rp, Tp = _wv.incoh_multilayer(
        wavelength,
        N_layers=[N_host_eff],
        thickness=thickness,
        aoi=theta_rad,
        N_above=N_above,
        N_below=N_below,
        polarization='TM',
    )
    Rs, Ts = _wv.incoh_multilayer(
        wavelength,
        N_layers=[N_host_eff],
        thickness=thickness,
        aoi=theta_rad,
        N_above=N_above,
        N_below=N_below,
        polarization='TE',
    )
    T    = (Ts + Tp)/2
    Rtot = (Rp + Rs)/2
    
    theta1 = _wv.snell(N_above, N_host_eff, theta_rad)
        
    Ttot = T*_np.exp(-k_abs*thickness/_np.cos(theta1.real))
    Tspec = T*_np.exp(-k_ext*thickness/_np.cos(theta1.real))
    Tdif = Ttot - Tspec

    # store data into a dataframe (λ index)
    results_df = _pd.DataFrame({
        'Rtot': Rtot,
        'Ttot': Ttot,
        'Tspec': Tspec,
        'Tdif': Tdif
    }, index=wavelength)

    results_df.index.name = 'Wavelength (µm)'

    return results_df

@_hide_signature
def adm_sphere(wavelength: _Union[float, _np.ndarray],                              # wavelengths [µm]
                N_host: _Union[float, _np.ndarray],                                     # host refractive index
                N_particle: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],  # particle refractive index
                D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],   # sphere diameters [µm] 
                fv: float,                                                          # film volume fraction 
                thickness: float,                                                       # film thickness [mm]       
                *,
                N_above: _Union[float, _np.ndarray] = 1.0,                          # refractive index above
                N_below: _Union[float, _np.ndarray] = 1.0,                          # refractive index below
                size_dist: _np.ndarray = None,                                      # number-fraction weights p_i 
                dependent_scatt = False,                                            # use Perkus-Yevik for dependent scattering
                effective_medium: bool = False,                                     # whether to compute effective N_host via Bruggeman
                use_phase_fun: bool = False,                                        # whether to use phase function instead of g
                n_theta: _Optional[int] = 101,                                          # auto|linspace|legendre angular sampling grid
                quad_pts: int = 16,                                                 # IAD quadrature points (also sizes the Legendre-moments phase representation)
                cone_incidence: _Optional[_Tuple[float, float]] = None,             # (theta_min, theta_max) in degrees for diffuse cone incidence
                lambertian: bool = False                                            # whether to compute Lambertian incidence instead of normal
                ):
    '''
    Parameters
    wavelength : array-like, shape (nλ,)
        Wavelengths [µm], strictly positive.

    N_host : float or array-like (nλ,)
        Host refractive index (can be complex). If array-like, length must equal len(wavelength).

    N_particle (float, 1darray or list): Complex refractive index of each 
                                            shell layer. N_particle.shape[1] == len(D). 
        Options are:
        float:   solid sphere and constant refractive index
        1darray: solid sphere and spectral refractive index (len = wavelength)
        list:    multilayered sphere (with both constant or spectral refractive indexes)
    
    D : float, _np.ndarray or list
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        if multilayer sphere, use list of floats (monodisperse) or arrays (polydisperse).
    
    fv : float
        Particle volume fraction in (0, 1). Used only to compute an effective medium N_host via
        `nk.emt_brugg(fv, N_particle, N_host)`.

    thickness : float
        Film thickness [mm], ≥ 0.
            
    N_above : float or array-like, optional
        Refractive index above the film. Default is 1.0 (air). If array-like, length must equal len(wavelength).

    N_below : float or array-like, optional
        Refractive index below the film. Default is 1.0 (air). If array-like, length must equal len(wavelength).

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
        When True, the phase function is passed to iadpython as Legendre moments
        (computed via Gauss-Legendre quadrature directly on the ensemble phase
        function -- no interpolation), sized for `quad_pts`.

    n_theta : int, optional
        Number of angular points used to sample the phase function and structure factor.
        Default is 101.

    quad_pts : int, optional
        iadpython adding-doubling quadrature points. Also determines how many
        Legendre moments are computed for the `use_phase_fun=True` path
        (2*quad_pts+1). Default is 16.

    cone_incidence : tuple, optional
        Optional tuple (theta_min, theta_max) in degrees for diffuse cone incidence    
    
    lambertian : bool, optional
        Whether to compute Lambertian incidence instead of normal incidence

    Returns
    - results_df : _pd.DataFrame with index=wavelength and columns:
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
    wavelength, N_host, N_particle, D, size_dist = _check_mie_inputs(wavelength, N_host, N_particle, D, size_dist=size_dist)

    n_wavelengths = wavelength.size
    N_above = _as_1d_array(N_above, "N_above", n_wavelengths=n_wavelengths, dtype=complex)
    N_below = _as_1d_array(N_below, "N_below", n_wavelengths=n_wavelengths, dtype=complex)

    if not (0 <= float(fv) < 1):
        raise ValueError("fv (volume fraction) must be in [0,1).")
    if not _np.isscalar(thickness) or thickness < 0:
        raise ValueError("thickness must be a nonnegative scalar in mm.")

    # if dependent scatt, check that particle is metallic through: Im(N_particle) > Re(N_particle)
    # if dependent_scatt:
    #     if _np.any(_np.array(N_particle).real < _np.array(N_particle).imag):
    #         print("Warning: Dependent scattering theory not recommended for metallic particles.")

    # ---------- Effective medium for host (if your convention is to dress N_host) ----------
    N_host_eff = N_host.copy()
    if effective_medium and fv > 0.0:
        N_host_eff = _effective_host(fv, N_particle, N_host, D, size_dist,
                                    emt_multilayer_fn=_emt_multilayer_sphere,
                                    emt_brugg_fn=_emt_brugg,
                                    )
   
   #  compute cross sections. Capture the per-size-bin Mie coefficient cache
   #  (out_coeff_cache) so a separate phase_function_moments call below can
   #  reuse it (an/bn depend only on size/wavelength, not scattering angle)
   #  instead of recomputing the Mie coefficient series from scratch.
    coeff_cache = []
    cabs, csca, gcos, _ = _mie.cross_section_ensemble(wavelength, N_host_eff, N_particle, D, fv,
                                                                size_dist=size_dist,
                                                                n_theta=n_theta,
                                                                check_inputs=False,
                                                                effective_medium=False,
                                                                dependent_scatt=dependent_scatt,
                                                                phase_function=False,
                                                                out_coeff_cache=coeff_cache)

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
        # Legendre moments computed directly via Gauss-Legendre quadrature on
        # mu=cos(theta) -- no interpolation -- reusing the Mie coefficient
        # cache captured above.
        a_l = _mie.phase_function_moments(wavelength, N_host_eff, N_particle, D, fv,
                                          size_dist=size_dist,
                                          n_moments=2*int(quad_pts) + 1,
                                          dependent_scatt=dependent_scatt,
                                          coeff_cache=coeff_cache,
                                          check_inputs=False)
        df_results = adm(wavelength, thickness, k_sca, k_abs, N_host=N_host_eff,
                                       phase_moments=a_l,
                                       N_above=N_above,
                                       N_below=N_below,
                                       quad_pts=quad_pts,
                                       cone_incidence=cone_incidence,
                                       lambertian=lambertian)
    else:
        df_results = adm(wavelength, thickness, k_sca, k_abs, N_host=N_host_eff,
                                       gcos=gcos,
                                       N_above=N_above,
                                       N_below=N_below,
                                       cone_incidence=cone_incidence,
                                       lambertian=lambertian)

    return df_results

@_hide_signature
def adm(wavelength, thickness, k_sca, k_abs, N_host,
        gcos=None,                                              # optional: asymmetry parameter per λ
        *,
        phase_fun=None,                                         # optional: phase function vs θ (DataFrame only; θ index in degrees 0..180)
        phase_moments=None,                                     # optional: Legendre moments a_l, shape (n_mom, nλ) -- bypasses TABULATED's interpolation
        N_above=1.0,                                                # refractive index above
        N_below=1.0,                                                # refractive index below
        quad_pts: int = 16,                                     # IAD quadrature points when using a tabulated PF
        cone_incidence: _Optional[_Tuple[float, float]] = None, # (theta_min, theta_max) in degrees for diffuse cone incidence
        lambertian: bool = False                                # whether to compute Lambertian incidence instead of normal
):
    """
    Adding–doubling (IAD) reflectance/transmittance for a scattering/absorbing film.

    IN_particleuts (arrays are per-wavelength unless noted):
    - wavelength : (nλ,) wavelengths [µm]
    - thickness    : scalar film thickness [mm]
    - k_sca    : (nλ,) scattering coefficient [µm^-1]
    - k_abs    : (nλ,) absorption coefficient [µm^-1]
    - N_host       : scalar or (nλ,) complex refractive index of the film host
    --------------------------------------------------------------------------
    Choose ONE angular description:
    - gcos     : (nλ,) asymmetry parameter  (Henyey–Greenstein style)
      OR
    - phase_fun: _pd.DataFrame of the differential *phase function* (not normalized to 1/4π),
                 shape (nθ, nλ). **Index must be θ in degrees from 0 to 180.**
                 Columns must be the wavelengths (same values as `wavelength`, order-agnostic).
                 The function will convert θ→μ=cosθ and sort μ ascending in [-1, 1].
                 Intended for external/measured tabulated data.
      OR
    - phase_moments: np.ndarray of Legendre moments a_l, shape (n_mom, nλ),
                 columns in the same order as `wavelength`. Bypasses the
                 spline+quadrature interpolation `phase_fun` requires --
                 use `empylib.miescattering.phase_function_moments` to
                 compute these directly from a Mie/Percus-Yevick ensemble.
    --------------------------------------------------------------------------
    - N_above, N_below : scalar or (nλ,) complex refractive indices above/below (defaults=1.0)
    - quad_pts : quadrature points for IAD when using a tabulated phase function
    - cone_incidence : optional tuple (theta_min, theta_max) in degrees for
                       diffuse cone incidence calculations

    Returns:
    - results_df : _pd.DataFrame with index=wavelength and columns:
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
    wavelength = _np.atleast_1d(_np.asarray(wavelength, float))
    k_sca = _np.atleast_1d(_np.asarray(k_sca, float))
    k_abs = _np.atleast_1d(_np.asarray(k_abs, float))

    if wavelength.ndim != 1:
        raise ValueError("wavelength must be a 1D array of wavelengths [µm].")
    n_wavelengths = wavelength.size
    for name, arr in [("k_sca", k_sca), ("k_abs", k_abs)]:
        if _np.asarray(arr).shape != (n_wavelengths,):
            raise ValueError(f"{name} must have the same length as wavelength.")

    N_host_arr  = _as_1d_array(N_host,  "N_host" , n_wavelengths=n_wavelengths, dtype=complex)
    N_above_arr = _as_1d_array(N_above, "N_above", n_wavelengths=n_wavelengths, dtype=complex)
    N_below_arr = _as_1d_array(N_below, "N_below", n_wavelengths=n_wavelengths, dtype=complex)

    # keep all positive AND finite. An overflowing Mie cross section (e.g. a very
    # large size parameter in the coated-sphere coefficient reconstruction) can
    # return +inf/NaN; left unchecked this propagates to a = mu_s/mu_t = inf/inf
    # = NaN and crashes the downstream iadpython solver. Map NaN -> 0 (no
    # interaction) and +inf -> a large-but-finite coefficient (optically opaque).
    _K_MAX = 1.0e6  # [µm^-1] far beyond any physical loading -> effectively opaque
    k_sca = _np.nan_to_num(_np.maximum(k_sca, 0.0), nan=0.0, posinf=_K_MAX, neginf=0.0)
    k_abs = _np.nan_to_num(_np.maximum(k_abs, 0.0), nan=0.0, posinf=_K_MAX, neginf=0.0)
    N_host_arr.imag = _np.maximum(N_host_arr.imag, 0.0)

    # ---------- convert to IAD units (mm^-1); include host absorption via Im(n) ----------
    # k_sca, k_abs are in µm^-1 -> mm^-1 multiply by 1e3
    mu_s = k_sca * 1e3

    # host material absorption: α_host = 4π k / λ  (λ in µm)  -> in mm^-1 multiply by 1e3
    kz_imag = 2.0 * _np.pi / wavelength * N_host_arr.imag * 1e3  # (2π/λ)*k in mm^-1
    mu_a = k_abs * 1e3 + 2.0 * kz_imag                # = (k_abs + 2*(2π/λ)k)*1e3 = (k_abs + 4πk/λ)*1e3
    mu_t = mu_s + mu_a
    d = float(thickness)

    # ---------- choose angular description ----------
    use_pf = phase_fun is not None
    use_moments = phase_moments is not None
    _n_given = sum((gcos is not None, use_pf, use_moments))
    if _n_given > 1:
        raise ValueError("Provide exactly one of gcos, phase_fun, or phase_moments.")
    if _n_given == 0:
        raise ValueError("You must provide gcos (per λ), a tabulated phase_fun, or phase_moments.")

    if not use_pf and not use_moments:
        gcos = _np.atleast_1d(_np.asarray(gcos, float))
        if gcos.shape != (n_wavelengths,):
            raise ValueError("gcos must have shape (len(wavelength),).")

    # ---------- prepare phase function (TABULATED path) ----------
    if use_pf:
        if not isinstance(phase_fun, _pd.DataFrame):
            raise TypeError("phase_fun must be a pandas DataFrame with theta-degree index in [0,180].")

        if _IAD_SUPPORTS_TABULATED_PF:
            # Shared conversion keeps theta->mu tabulation consistent across
            # all future iadpython call-sites.
            pf_df = _prepare_tabulated_phase_fun_for_iad(phase_fun, wavelength)

            # Sanitize non-finite phase-function values that can arise from Mie
            # size-parameter overflow (e.g. large coated-sphere recursion at UV
            # wavelengths). CubicSpline inside iadpython.redistribution rejects
            # inf/NaN; replace entire bad wavelength-columns with the isotropic
            # phase function (p = 1.0, equivalent to g = 0). The optimizer sees a
            # large residual at those wavelengths and steers away from the
            # pathological parameters — the same logic as the k_sca/k_abs clamp
            # applied above.
            _pf_vals = pf_df.values
            _bad_cols = ~_np.all(_np.isfinite(_pf_vals), axis=0)
            if _bad_cols.any():
                _pf_vals = _pf_vals.copy()
                _pf_vals[:, _bad_cols] = 1.0   # isotropic fallback (∫p dμ = 2 satisfied)
                pf_df = _pd.DataFrame(_pf_vals, index=pf_df.index, columns=pf_df.columns)
        else:
            # Older iadpython releases only expose Henyey-Greenstein anisotropy.
            # Collapse the supplied phase function to an equivalent g(λ) so the
            # public `phase_fun=` entrypoint remains usable in those environments.
            _, gcos = _mie.scatter_from_phase_function(phase_fun)
            use_pf = False

    # ---------- prepare phase function (MOMENTS path) ----------
    if use_moments:
        phase_moments = _np.asarray(phase_moments, dtype=float)
        if phase_moments.ndim != 2 or phase_moments.shape[1] != n_wavelengths:
            raise ValueError(
                "phase_moments must have shape (n_mom, len(wavelength)); "
                f"got {phase_moments.shape} for {n_wavelengths} wavelengths."
            )
        # Sanitize non-finite moment columns the same way the TABULATED path
        # sanitizes non-finite phase-function values (Mie size-parameter
        # overflow at extreme wavelengths): replace the entire column with
        # the isotropic moments (a_0=1, a_l=0 for l>0, equivalent to g=0).
        _bad_cols = ~_np.all(_np.isfinite(phase_moments), axis=0)
        if _bad_cols.any():
            phase_moments = phase_moments.copy()
            phase_moments[:, _bad_cols] = 0.0
            phase_moments[0, _bad_cols] = 1.0

    # ---------- run IAD per wavelength ----------
    Ttot  = _np.zeros(n_wavelengths, float)
    Rtot  = _np.zeros(n_wavelengths, float)
    Tspec = _np.zeros(n_wavelengths, float)
    Rspec = _np.zeros(n_wavelengths, float)

    if cone_incidence is not None:
        Tcone = _np.zeros(n_wavelengths, float)
        Rcone = _np.zeros(n_wavelengths, float)

    if lambertian:
        Rlam = _np.zeros(n_wavelengths, float)
        Tlam = _np.zeros(n_wavelengths, float)

    for j in range(n_wavelengths):
        n_transport = float(N_host_arr.real[j])
        if n_transport <= 0.0:
            raise ValueError("N_host real part must be strictly positive for radiative transport.")

        if mu_t[j] <= 0.0:
            a = 0.0
            b = 0.0
        else:
            a = float(mu_s[j] / mu_t[j])   # single-scattering albedo
            b = float(mu_t[j] * d)         # optical thickness

        sample_kwargs = dict(
            a=a,
            b=b,
            d=d,
            n=n_transport,
            n_sample_boundary=complex(N_host_arr[j]),
            n_above=1.0,
            n_below=1.0,
            n_outer_above=complex(N_above_arr[j]),
            n_outer_below=complex(N_below_arr[j]),
        )
        if use_moments:
            # direct Legendre-moments path (bypasses TABULATED's interpolation)
            s = _iad.Sample(
                **sample_kwargs,
                quad_pts=int(quad_pts),
                pf_type="MOMENTS",
                pf_data=phase_moments[:, j],
            )
        elif not use_pf:
            sample_kwargs["g"] = float(_np.nan_to_num(gcos[j], nan=0.0))
            s = _iad.Sample(**sample_kwargs, quad_pts=int(quad_pts))
        else:
            # tabulated phase function path
            pf_col = pf_df.iloc[:, j].to_frame()
            s = _iad.Sample(
                **sample_kwargs,
                quad_pts=int(quad_pts),
                pf_type="TABULATED",
                pf_data=pf_col,
            )

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

    # Make small values == 0
    Rdif[(_np.abs(Rdif) < 1E-2) & (Rdif < 0)] = 0
    Tdif[(_np.abs(Tdif) < 1E-2) & (Tdif < 0)] = 0

    # Store data into a dataframe (λ index)
    results_df = _pd.DataFrame({
        "Rtot": Rtot,
        "Ttot": Ttot,
        "Rspec": Rspec,
        "Tspec": Tspec,
        "Rdif": Rdif,
        "Tdif": Tdif
    }, index=wavelength)

    results_df.index.name = 'Wavelength (µm)'

    if cone_incidence is not None:
        results_df["Rcone"] = Rcone
        results_df["Tcone"] = Tcone
    
    if lambertian:
        results_df["Rlam"] = Rlam
        results_df["Tlam"] = Tlam

    return results_df
