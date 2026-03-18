# -*- coding: utf-8 -*-
"""
Library of wave optics funcions

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""

import numpy as np
from .utils import _as_1d_array, _as_carray, _as_real_array, _normalize_multilayer_inputs


def _resolve_polarization(polarization):
    """Normalize polarization input.

    Returns
    -------
    tuple
        ('TE', 'TM') for unpolarized input (False or None), or a single-element
        tuple containing the requested polarization.
    """
    if polarization is False or polarization is None:
        return ('TE', 'TM')

    if isinstance(polarization, str):
        normalized = polarization.upper()
        if normalized in ('TE', 'TM'):
            return (normalized,)

    raise ValueError("polarization must be False, 'TE', or 'TM'")


def _unpolarized_output(results_TE, results_TM):
    '''Average TE and TM results for unpolarized output format.'''

    if len(results_TE) != len(results_TM):
        raise ValueError('TE and TM results must have the same number of outputs')
    
    # average R and T for unpolarized output
    R_TE, T_TE = results_TE[:2]
    R_TM, T_TM = results_TM[:2]
    R = (R_TE + R_TM) / 2
    T = (T_TE + T_TM) / 2

    # simulation from incoherent sources may only have R and T
    if len(results_TE) == 2:
        out = (R, T)

    # simulation from coherent sources also have r and t
    elif len(results_TE) == 4:
        r_TE, t_TE = results_TE[2:]
        r_TM, t_TM = results_TM[2:]
        r = {'TE': r_TE, 'TM': r_TM}
        t = {'TE': t_TE, 'TM': t_TM}

        out = (R, T, r, t)
    
    return out

def interface(N_above, N_below, *, aoi = 0, polarization=False):
    '''
    Computes the Fresnel coeficients  and energy flux of at an interface. For
    each angle of incidence, this function will compute the Fresnel coeficients at the 
    spectrum defined for N_above or N_below.
    
    Parameters
    ----------
    N_above : ndarray or float
        Spectral refractive index medium above the interface.
        
    N_below : ndarray or float
        Spectral refractive index medium below the interface.
        
    aoi : ndarray or float
        Angle of incidence in radians (real or complex).
        
    polarization: str (optional)
        Polarization of incident field. Could be:
            - False for unpolarized light (default, averages TE and TM)
            - 'TM' transverse magnetic
            - 'TE' transverse electric

    Returns
    -------
    R: ndarray
        Reflectivity
        
    T: ndarray
        Transmissivity
        
    r : ndarray
        Reflection coeficient
        
    t: ndarray
        Transmission coeficient
    
    '''
    # Normalize inputs to 1D spectral/angle arrays.
    th = _as_1d_array(aoi, 'aoi', dtype=complex)
    th_scalar = np.isscalar(aoi)

    n1_raw = np.asarray(N_above)
    n2_raw = np.asarray(N_below)
    ns = max(n1_raw.size if n1_raw.ndim else 1, n2_raw.size if n2_raw.ndim else 1)
    n1 = _as_carray(N_above, 'N_above', ns, val_type=complex)
    n2 = _as_carray(N_below, 'N_below', ns, val_type=complex)

    # Build angle x wavelength grids for vectorized Fresnel computation.
    n_i, th_grid = np.meshgrid(n1, th)
    n_t = np.meshgrid(n2, th)[0]

    sin_i = np.sin(th_grid)
    cos_i = np.cos(th_grid)
    sin_t = n_i * sin_i / n_t
    cos_t = np.sqrt(1 - sin_t**2)

    # Determine which polarization(s) to compute based on input.
    pols = _resolve_polarization(polarization)

    # Fresnel amplitude coefficients (r, t) and power coefficients (R, T).
    out = {}
    for p in pols:
        if p == 'TM':
            r = (n_i * cos_t - n_t * cos_i) / (n_i * cos_t + n_t * cos_i)
            t = (2 * n_i * cos_i) / (n_i * cos_t + n_t * cos_i)
            R = np.abs(r * np.conj(r))
            T = np.abs(
                np.real(np.conj(n_t) * cos_t) * t * np.conj(t)
                / np.real(np.conj(n_i) * cos_i)
            )
        elif p == 'TE':
            r = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)
            t = (2 * n_i * cos_i) / (n_i * cos_i + n_t * cos_t)
            R = np.abs(r * np.conj(r))
            T = np.abs(
                np.real(n_t * cos_t) * t * np.conj(t)
                / np.real(n_i * cos_i)
            )

        if th_scalar:
            out[p] = (
                R.reshape(-1,),
                T.reshape(-1,),
                r.reshape(-1,),
                t.reshape(-1,),
            )
        else:
            out[p] = (R, T, r, t)

    if len(pols) == 1:
        return out[pols[0]]
    return _unpolarized_output(out['TE'], out['TM'])

def multilayer(
    wavelength,
    N_layers=None,
    thickness=None,
    *,
    aoi=0,
    N_above=1.0,
    N_below=1.0,
    polarization=False
):
    '''
    Get Fresnel coeficients and energy flux of multilayered films. The function 
    computes the spectral Fresnel coefficients at each angle of incidence

    Parameters
    ----------
    wavelength : ndarray or float
        Wavelength range in microns.
        
    N_layers : float, ndarray, or list
        Refractive index of each finite layer. It can be:
        - a single float (single-layer, wavelength-independent),
        - a single 1D ndarray (single-layer, wavelength-dependent),
        - a list of floats,
        - or a mixed list of floats and 1D ndarrays, where each ndarray has
          size len(wavelength).
        
    thickness : list or float
        Thickness of each layer in microns. A single float is allowed for a
        single-layer case.
        
    aoi : ndarray or float, optional
        Angle of incidence in radians.

    N_above : float (optional)
        Refractive index of the medium above the film (default 1.0)
    
    N_below : float (optional)
        Refractive index of the medium below the film (default 1.0)
        
    polarization: bool or str (optional)
        Polarization of incident field:
            - False for unpolarized light (default, averages TE and TM)
            - 'TM' transverse magnetic
            - 'TE' transverse electric

    Returns
    -------
    R: ndarray
        Reflectivity
        
    T: ndarray
        Transmissivity
        
    r : ndarray
        Reflection coeficient
        
    t: ndarray
        Transmission coeficient

    Notes
    -----
    If ``N_layers`` is ``None`` or empty, the computation falls back to a
    single-interface Fresnel evaluation between ``N_above`` and ``N_below``.

    '''
    if thickness is None:
        thickness = []

    # Gracefully handle the no-layer case as a single interface calculation.
    if N_layers is None or (
        isinstance(N_layers, (list, tuple, np.ndarray)) and len(N_layers) == 0
    ):
        _, _, _, n0, nS = _assert_multilayer_input(
            wavelength,
            [],
            [],
            N_above=N_above,
            N_below=N_below,
        )
        return interface(n0, nS, aoi=aoi, polarization=polarization)

    # Validate and standardize multilayer inputs to spectral arrays.
    wl, dL, nL, n0, nS = _assert_multilayer_input(
        wavelength, thickness, N_layers, N_above=N_above, N_below=N_below
    )
    th = _as_1d_array(aoi, 'aoi', dtype=complex)
    th_scalar = np.isscalar(aoi)

    # Vacuum impedance and free-space wavenumber.
    z0 = 376.730
    k0 = 2 * np.pi / wl

    k_grid, th_grid = np.meshgrid(k0, th)
    n0_grid = np.meshgrid(n0, th)[0]
    nS_grid = np.meshgrid(nS, th)[0]

    s0 = np.sin(th_grid)
    c0 = np.sqrt(1 - s0**2) + 1e-15

    sS = n0_grid * s0 / nS_grid
    cS = np.sqrt(1 - sS**2) + 1e-15

    # Resolve polarization mode: TE, TM, or both for unpolarized light.
    pols = _resolve_polarization(polarization)

    q0_map = {}
    qS_map = {}
    tf_map = {}
    M11_map = {}
    M12_map = {}
    M21_map = {}
    M22_map = {}

    # Initialize interface admittances and transfer matrix per polarization.
    for p in pols:
        if p == 'TE':
            q0_map[p] = -z0 / (n0_grid * c0)
            qS_map[p] = -z0 / (nS_grid * cS)
            tf_map[p] = 1
        else:
            q0_map[p] = z0 * c0 / n0_grid
            qS_map[p] = z0 * cS / nS_grid
            tf_map[p] = c0 / cS

        M11_map[p] = 1
        M12_map[p] = 0
        M21_map[p] = 0
        M22_map[p] = 1

    # Multiply characteristic matrix of each finite layer.
    for n_j, d_j in zip(nL, dL):
        n_j_grid = np.meshgrid(n_j, th)[0]
        s_j = n0_grid * s0 / n_j_grid
        c_j = np.sqrt(1 - s_j**2) + 1e-10
        phi = n_j_grid * k_grid * c_j * d_j

        for p in pols:
            if p == 'TE':
                q_j = -z0 / (n_j_grid * c_j)
            else:
                q_j = z0 * c_j / n_j_grid

            M11 = M11_map[p]
            M12 = M12_map[p]
            M21 = M21_map[p]
            M22 = M22_map[p]

            N11 = np.cos(phi) * M11 - 1j / q_j * np.sin(phi) * M12
            N12 = -1j * q_j * np.sin(phi) * M11 + np.cos(phi) * M12
            N21 = np.cos(phi) * M21 - 1j / q_j * np.sin(phi) * M22
            N22 = -1j * q_j * np.sin(phi) * M21 + np.cos(phi) * M22

            M11_map[p] = N11
            M12_map[p] = N12
            M21_map[p] = N21
            M22_map[p] = N22

    # Extract effective global r, t and convert to R, T.
    out = {}
    for p in pols:
        q0 = q0_map[p]
        qS = qS_map[p]
        tf = tf_map[p]
        M11 = M11_map[p]
        M12 = M12_map[p]
        M21 = M21_map[p]
        M22 = M22_map[p]

        nr = (qS * M11 + M12) - (qS * M21 + M22) * q0
        dr = (qS * M11 + M12) + (qS * M21 + M22) * q0
        r = nr / dr
        t = (2 * tf * qS) / dr

        R = np.abs(r) ** 2
        if p == 'TE':
            T = np.real(nS_grid * cS) / np.real(n0_grid * c0) * np.abs(t) ** 2
        else:
            T = np.real(np.conj(nS_grid) * cS) / np.real(np.conj(n0_grid) * c0) * np.abs(t) ** 2

        if th_scalar:
            out[p] = (
                R.reshape(-1,),
                T.reshape(-1,),
                r.reshape(-1,),
                t.reshape(-1,),
            )
        else:
            out[p] = (R, T, r, t)

    if len(pols) == 1:
        return out[pols[0]]
    return _unpolarized_output(out['TE'], out['TM'])

def incoh_multilayer(
    wavelength,
    N_layers=None,
    thickness=None,
    *,
    aoi=0,
    N_above=1.0,
    N_below=1.0,
    polarization=False,
    coh_length=0
):
    '''
    Transfer matrix method (TMM) for coherent, and incoherent multilayer 
    structures. (Only tested at normal incidence)
    
    Source: Katsidis, C. C. & Siapkas, D. I. Appl. Opt. 41, 3978 (2002).

    Parameters
    ----------
    wavelength : ndarray
        wavelength range (microns)
        
    aoi : float, optional
        angle of incidence (radian).
        
    N_layers : float, ndarray, or list
        Refractive index of each finite layer. It can be:
        - a single float (single-layer, wavelength-independent),
        - a single 1D ndarray (single-layer, wavelength-dependent),
        - a list of floats,
        - or a mixed list of floats and 1D ndarrays, where each ndarray has
          size len(wavelength).
        
    thickness : list or float
        Thickness of each layer in microns. A single float is allowed for a
        single-layer case.

    N_above : float or ndarray, optional
        Refractive index of the medium above the multilayer film.

    N_below : float or ndarray, optional
        Refractive index of the medium below the multilayer film.
        
    polarization: bool or str (optional)
        Polarization of incident field:
            - False for unpolarized light (default, averages TE and TM)
            - 'TM' transverse magnetic
            - 'TE' transverse electric
            
    coh_length: float
        Coherence length of the source (microns), 0 by default

    Returns
    -------
    R: ndarray
        Reflectivity
        
    T: ndarray
        Transmissivity

    '''
    if N_layers is None:
        N_layers = []
    if thickness is None:
        thickness = []

    # Resolve polarization mode and validate all spectral inputs.
    pols = _resolve_polarization(polarization)
    wl, dL, nL, n0, nS = _assert_multilayer_input(
        wavelength, thickness, N_layers, N_above=N_above, N_below=N_below
    )

    if not np.isscalar(aoi):
        raise ValueError('aoi must be a scalar for incoh_multilayer')
    if coh_length < 0:
        raise ValueError('coh_length must be non-negative')

    th0 = float(aoi)
    lc = float(coh_length)

    # Mark layers as incoherent when optical path exceeds coherence length.
    d_arr = np.asarray(dL, dtype=float)
    inc_mask = d_arr > lc * np.cos(th0) / 2

    n_all = [n0] + nL + [nS]
    seg_n = [n_all[0]]
    seg_d = []

    # Initialize intensity-transfer matrices per polarization.
    T11_map = {p: 1 for p in pols}
    T12_map = {p: 0 for p in pols}
    T21_map = {p: 0 for p in pols}
    T22_map = {p: 1 for p in pols}

    th_start = th0 * np.ones(len(wl))
    # Traverse layers, splitting into coherent sub-stacks between incoherent layers.
    for j, is_inc in enumerate(inc_mask):
        if is_inc:
            seg_n.append(n_all[j + 1])
            coh = {}
            for p in pols:
                coh[p] = _TMMcoh(wl, th_start, seg_n, seg_d, p)

            th_end = coh[pols[0]][4]

            # In incoherent layers, only attenuation survives (random phase averaging).
            phi_d = (2* np.pi / wl * n_all[j + 1] * np.cos(th_end) * dL[j])
            att = np.exp(-2 * phi_d.imag)
            att = att * (att >= 1e-30) + (att < 1e-30) * 1e-30

            P11 = 1 / att
            P22 = att

            for p in pols:
                t11, t12, t21, t22, _ = coh[p]
                S11 = T11_map[p] * t11 + T12_map[p] * t21
                S12 = T11_map[p] * t12 + T12_map[p] * t22
                S21 = T21_map[p] * t11 + T22_map[p] * t21
                S22 = T21_map[p] * t12 + T22_map[p] * t22

                T11_map[p] = S11 * P11
                T12_map[p] = S12 * P22
                T21_map[p] = S21 * P11
                T22_map[p] = S22 * P22

            th_start = th_end
            seg_n = [n_all[j + 1]]
            seg_d = []
        else:
            seg_n.append(n_all[j + 1])
            seg_d.append(dL[j])

    seg_n.append(n_all[-1])
    coh = {}
    for p in pols:
        coh[p] = _TMMcoh(wl, th_start, seg_n, seg_d, p)

    th_end = coh[pols[0]][4]
    # Final coherent segment and total R, T per polarization.
    out = {}
    for p in pols:
        t11, t12, t21, t22, _ = coh[p]

        A11 = T11_map[p] * t11 + T12_map[p] * t21
        A21 = T21_map[p] * t11 + T22_map[p] * t21

        R = A21 / A11
        T = 1 / A11

        if p == 'TE':
            T = np.real(n_all[-1] * np.cos(th_end)) / np.real(n_all[0] * np.cos(th0)) * T
        else:
            T = (
                np.real(np.conj(n_all[-1]) * np.cos(th_end))
                / np.real(np.conj(n_all[0]) * np.cos(th0))
                * T
            )

        out[p] = (R, T)

    if len(pols) == 1:
        return out[pols[0]]
    return _unpolarized_output(out['TE'], out['TM'])

def _assert_multilayer_input(wavelength, thickness, N_layers, *, N_above=1.0, N_below=1.0):
    '''
    Verify that multilayer input complies with required dimensions

    Parameters
    ----------
    wavelength : float or ndarray
        wavelength range (microns)
        
    thickness : list or float
        Thickness of each finite layer in microns. Must contain float values.
        A single float is allowed for a single-layer case.
        
    N_layers : float, ndarray, or list
        Refractive index of each finite layer. It can be:
        - a single float (single-layer, wavelength-independent),
        - a single 1D ndarray (single-layer, wavelength-dependent),
        - a list of floats,
        - or a mixed list of floats and 1D ndarrays, where each ndarray has
          size len(wavelength).

    N_above : float or ndarray, optional
        Refractive index of the semi-infinite medium above.

    N_below : float or ndarray, optional
        Refractive index of the semi-infinite medium below.

    Returns
    -------
    wavelength : ndarray
        Wavelength range (microns).

    thicknesses : list
        Thickness of each layer in microns.

    N_layers : list
        Finite-layer refractive indices as ndarrays of size len(wavelength).

    N_above_arr : ndarray
        Upper semi-infinite medium refractive index as ndarray of size len(wavelength).

    N_below_arr : ndarray
        Lower semi-infinite medium refractive index as ndarray of size len(wavelength).

    '''
    # Wavelength must be positive and represented as 1D array.
    return _normalize_multilayer_inputs(
        wavelength,
        thickness,
        N_layers,
        N_above=N_above,
        N_below=N_below,
    )

def _TMMcoh(wavelength, aoi, N_layers, thickness, polarization):
    '''
    Calculate the coherent transfer matrix for a multilayer stack.
    Parameters
    ----------
    wavelength : ndarray
        wavelength range (microns)
    aoi : ndarray
        angle of incidence for each wavelength (radians)
    N_layers : list of ndarrays
        refractive indices of each layer for each wavelength
    thickness : list of floats
        thickness of each layer (microns)
    polarization : str
        polarization of the incident light ('s' or 'p')
        
    Returns
    -------
    T11_coh : ndarray
        Coherent transfer matrix element T11 for forward incidence.
    T12_coh : ndarray
        Coherent transfer matrix element T12 for forward incidence.
    T21_coh : ndarray
        Coherent transfer matrix element T21 for forward incidence.
    T22_coh : ndarray
        Coherent transfer matrix element T22 for forward incidence.
    th_end : ndarray
        Angle of the final medium for each wavelength after Snell's law propagation.
    '''
    # Build forward/backward coherent stacks for intensity transfer coefficients.
    d_f = thickness.copy()
    d_b = thickness.copy()
    d_b.reverse()
    
    # Iterate wavelength-by-wavelength to evaluate forward and reverse stacks.
    th_end = np.zeros(len(wavelength), dtype=complex)
    r_f = np.zeros(len(wavelength), dtype=complex)
    t_f = np.zeros(len(wavelength), dtype=complex)
    r_b = np.zeros(len(wavelength), dtype=complex)
    t_b = np.zeros(len(wavelength), dtype=complex)

    for i in range(len(wavelength)):
        # Snell propagation to final medium angle for this wavelength.
        th_end[i] = snell(
            N_layers[0][i],
            N_layers[-1][i],
            aoi[i],
        )

        n_f = []
        n_b = []
        for j in range(len(N_layers)):
            n_f.append(N_layers[j][i])
            n_b.append(N_layers[-(j + 1)][i])

        # Coherent amplitudes for forward and reverse incidence.
        r_f[i], t_f[i] = multilayer(
            wavelength[i],
            aoi=aoi[i],
            N_layers=n_f[1:-1],
            thickness=d_f,
            N_above=n_f[0],
            N_below=n_f[-1],
            polarization=polarization
        )[-2:]
        r_b[i], t_b[i] = multilayer(
            wavelength[i],
            aoi=th_end[i],
            N_layers=n_b[1:-1],
            thickness=d_b,
            N_above=n_b[0],
            N_below=n_b[-1],
            polarization=polarization
        )[-2:]

    T11_coh = 1 / np.abs(t_f) ** 2
    T12_coh = -np.abs(r_b) ** 2 / np.abs(t_f) ** 2
    T21_coh = np.abs(r_f) ** 2 / np.abs(t_f) ** 2
    T22_coh = (
        np.abs(t_f * t_b) ** 2 - np.abs(r_f * r_b) ** 2
    ) / np.abs(t_f) ** 2
    return T11_coh, T12_coh, T21_coh, T22_coh, th_end


#------------------------------------------------------------------
# These are extra functions from TMM python code by Steve Byrnes
# source https://github.com/sbyrnes321/tmm
#------------------------------------------------------------------
def _is_forward_angle(n, theta):
    """
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.
    """
    from numpy import cos, abs
    import sys
    EPSILON = sys.float_info.epsilon # typical floating-point calculation error


    assert n.real * n.imag >= 0, ("For materials with gain, it's ambiguous which "
                                  "beam is incoming vs outgoing. See "
                                  "https://arxiv.org/abs/1603.02720 Appendix C.\n"
                                  "n: " + str(n) + "   angle: " + str(theta))
    ncostheta = n * cos(theta)
    if abs(ncostheta.imag) > 100 * EPSILON:
        # Either evanescent decay or lossy medium. Either way, the one that
        # decays is the forward-moving wave
        answer = (ncostheta.imag > 0)
    else:
        # Forward is the one with positive Poynting vector
        # Poynting vector is Re[n cos(theta)] for s-polarization or
        # Re[n cos(theta*)] for p-polarization, but it turns out they're consistent
        # so I'll just assume s then check both below
        answer = (ncostheta.real > 0)
    # convert from numpy boolean to the normal Python boolean
    answer = bool(answer)
    # double-check the answer ... can't be too careful!
    error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
                    " index maybe?\n"
                    "n: " + str(n) + "   angle: " + str(theta))
    if answer is True:
        assert ncostheta.imag > -100 * EPSILON, error_string
        assert ncostheta.real > -100 * EPSILON, error_string
        assert (n * cos(theta.conjugate())).real > -100 * EPSILON, error_string
    else:
        assert ncostheta.imag < 100 * EPSILON, error_string
        assert ncostheta.real < 100 * EPSILON, error_string
        assert (n * cos(theta.conjugate())).real < 100 * EPSILON, error_string
    return answer

@np.vectorize
def snell(n_1, n_2, th_1):
    """
    return angle theta in layer 2 with refractive index n_2, assuming
    it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
    that "angles" may be complex!!
    """
    from numpy.lib.scimath import arcsin
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    th_2_guess = arcsin(n_1*np.sin(th_1) / n_2)
    if _is_forward_angle(n_2, th_2_guess):
        return th_2_guess
    else:
        return np.pi - th_2_guess

def _list_snell(n_list, th_0):
    """
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
    """
    from numpy.lib.scimath import arcsin
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    angles = arcsin(n_list[0]*np.sin(th_0) / n_list)
    # The first and last entry need to be the forward angle (the intermediate
    # layers don't matter, see https://arxiv.org/abs/1603.02720 Section 5)
    if not _is_forward_angle(n_list[0], angles[0]):
        angles[0] = np.pi - angles[0]
    if not _is_forward_angle(n_list[-1], angles[-1]):
        angles[-1] = np.pi - angles[-1]
    return angles
