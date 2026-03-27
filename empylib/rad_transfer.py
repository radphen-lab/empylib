# -*- coding: utf-8 -*-
"""
Library of radiative transfer function

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import numpy as _np
from . import miescattering as _mie
from . import waveoptics as _wv
from . import nklib as _nk
import iadpython as _iad
import pandas as _pd
from typing import Union as _Union, Optional as _Optional, List as _List, Tuple as _Tuple
from .utils import _as_1d_array, _check_mie_inputs, _hide_signature
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
        N_particle_eff = _emt_multilayer_sphere(D_layers_mean, N_particle, check_inputs=False)
        N_host = _emt_brugg(fv, N_particle_eff, N_host)

    # ---------- Mie cross sections and phase function ----------
    cabs, csca, _, _ = _mie.cross_section_ensemble(wavelength, N_host, N_particle, D, fv, 
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
        N_layers=[N_host],
        thickness=thickness,
        aoi=theta_rad,
        N_above=N_above,
        N_below=N_below,
        polarization='TM',
    )
    Rs, Ts = _wv.incoh_multilayer(
        wavelength,
        N_layers=[N_host],
        thickness=thickness,
        aoi=theta_rad,
        N_above=N_above,
        N_below=N_below,
        polarization='TE',
    )
    T    = (Ts + Tp)/2
    Rtot = (Rp + Rs)/2
    
    theta1 = _wv.snell(N_above, N_host, theta_rad)
        
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
        Using the phase function is more accurate but also more computationally intensive.

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
    N_layers = len(D)                                    # number of layers in the sphere
    
    N_host_eff = N_host.copy()
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
        N_particle_eff = _emt_multilayer_sphere(D_layers_mean, N_particle, check_inputs=False)
        N_host_eff = _emt_brugg(fv, N_particle_eff, N_host)
    
    # ---------- Mie cross sections and phase function ----------
    theta_eval = _np.linspace(0, _np.pi, 100)
    cabs, csca, gcos, phase_scatter = _mie.cross_section_ensemble(wavelength, N_host_eff, N_particle, D, fv, 
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
        df_results = adm(wavelength, thickness, k_sca, k_abs, N_host=N_host_eff, 
                                       phase_fun=phase_scatter, 
                                       N_above=N_above, 
                                       N_below=N_below,
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

    # keep all positive
    k_sca = _np.maximum(k_sca, 0.0)
    k_abs = _np.maximum(k_abs, 0.0)
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
    if use_pf and (gcos is not None):
        raise ValueError("Provide either gcos OR phase_fun, not both.")
    if (not use_pf) and (gcos is None):
        raise ValueError("You must provide gcos (per λ) or a tabulated phase_fun.")

    if not use_pf:
        gcos = _np.atleast_1d(_np.asarray(gcos, float))
        if gcos.shape != (n_wavelengths,):
            raise ValueError("gcos must have shape (len(wavelength),).")

    # ---------- prepare phase function (TABULATED path) ----------
    if use_pf:
        if not isinstance(phase_fun, _pd.DataFrame):
            raise TypeError("phase_fun must be a pandas DataFrame with θ-degree index in [0,180].")

        # Validate θ index: must be degrees from 0 to 180
        theta_idx = _np.asarray(phase_fun.index, float)
        if theta_idx.ndim != 1:
            raise ValueError("phase_fun index must be 1D θ (degrees).")
        if theta_idx.min() < 0.0 or theta_idx.max() > 180.0:
            raise ValueError("phase_fun index (θ) must lie within [0°, 180°].")

        # Align columns to wavelength (order-agnostic but values must match)
        try:
            # If columns are numeric wavelengths, reindex to wavelength exactly (no interpolation here)
            PF = phase_fun.reindex(columns=wavelength, copy=False).values
        except Exception as e:
            raise ValueError("phase_fun columns must match wavelength.") from e
        if PF.shape != (theta_idx.size, n_wavelengths):
            raise ValueError("phase_fun must have shape (nθ, nλ) matching wavelength.")

        if _IAD_SUPPORTS_TABULATED_PF:
            # Convert θ → μ and sort ascending in [-1, 1]
            mu = _np.cos(_np.radians(theta_idx))
            order = _np.argsort(mu)    # ascending
            mu = mu[order]
            PF = PF[order, :]

            # IAD expects a DataFrame with index μ in [-1,1], one column per λ
            pf_df = _pd.DataFrame(PF, index=mu, columns=wavelength)
            pf_df.index.name = "cos(theta)"
        else:
            # Older iadpython releases only expose Henyey-Greenstein anisotropy.
            # Collapse the supplied phase function to an equivalent g(λ) so the
            # public `phase_fun=` entrypoint remains usable in those environments.
            _, gcos = _mie.scatter_from_phase_function(phase_fun)
            use_pf = False

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
        # guard: IAD wants n >= 1
        n_real = float(max(N_host_arr.real[j], 1.0))
        n_up   = float(max(N_above_arr.real[j], 1.0))
        n_dw   = float(max(N_below_arr.real[j], 1.0))

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
