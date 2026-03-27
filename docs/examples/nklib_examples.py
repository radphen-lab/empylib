from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import warnings

from .bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import empylib.nklib as nk
from scipy.integrate import IntegrationWarning


def local_material_demo():
    lam = np.linspace(0.40, 2.00, 250)
    n_sio2, nk_table = nk.get_nkfile(
        lam,
        "sio2_Palik_Lemarchand2013",
        get_from_local_path=True,
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, n_sio2.real, label="n")
    ax.plot(lam, n_sio2.imag, label="k")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Optical constants")
    ax.set_title("Silica loaded from a local .nk file")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "nk": n_sio2, "table": nk_table, "fig": fig}


def oscillator_demo():
    lam = np.linspace(0.45, 1.80, 220)
    oscillator_dict = {
        "lorentz_1": {"type": "lorentz", "epsinf": 1.0, "wp": 7.5, "wn": 3.0, "gamma": 0.25},
        "gaussian_1": {"type": "gaussian", "A": 1.1, "Br": 0.30, "E0": 2.0},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        n_mix = nk.multi_oscillator(lam, oscillator_dict)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, n_mix.real, label="n")
    ax.plot(lam, n_mix.imag, label="k")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Optical constants")
    ax.set_title("Combined oscillator model")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "nk": n_mix, "oscillator_dict": oscillator_dict, "fig": fig}


def fit_demo():
    lam = np.linspace(0.45, 1.80, 120)
    target = nk.multi_oscillator(
        lam,
        {"lorentz_1": {"type": "lorentz", "epsinf": 1.0, "wp": 7.0, "wn": 2.8, "gamma": 0.22}},
    )
    initial_guess = {
        "lorentz_1": {"type": "lorentz", "epsinf": 1.2, "wp": 6.5, "wn": 2.7, "gamma": 0.30},
    }
    fitted_oscillator, result = nk.fit_to_oscillator(
        lam,
        [target.real, target.imag],
        initial_guess,
    )

    fitted = nk.multi_oscillator(lam, fitted_oscillator.model)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, target.real, label="Target n")
    ax.plot(lam, fitted.real, "--", label="Fit n")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Refractive index")
    ax.set_title("Oscillator fit to synthetic data")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {
        "lam": lam,
        "target": target,
        "fitted": fitted,
        "fitted_oscillator": fitted_oscillator,
        "result": result,
        "fig": fig,
    }


def effective_medium_demo():
    lam = np.linspace(0.60, 2.20, 180)
    n_eff_layered = nk.emt_multilayer_sphere(
        [0.08, 0.10],
        [1.70 + 0.00j, 2.30 + 0.02j],
    )
    n_eff_mix = nk.emt_brugg(0.25, 2.4 + 0.10j, 1.5 + 0.00j)
    eps_imag = np.exp(-((lam - 1.20) / 0.25) ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        eps_real = nk.eps_real_kkr(lam, eps_imag, eps_inf=1.8)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, eps_real, label="KK reconstructed eps'")
    ax.axhline(np.real(n_eff_mix**2), color="tab:orange", linestyle="--", label="Bruggeman eps'")
    ax.axhline(np.real(n_eff_layered**2), color="tab:green", linestyle=":", label="Layered EMT eps'")
    ax.set_xlabel("Wavelength (um)")
    ax.set_title("Effective-medium and KK utilities")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {
        "lam": lam,
        "n_eff_layered": n_eff_layered,
        "n_eff_mix": n_eff_mix,
        "eps_real": eps_real,
        "fig": fig,
    }


def local_shortcuts_demo():
    lam = np.linspace(0.40, 1.50, 180)
    materials = {
        "CaCO3": nk.CaCO3(lam),
        "BaSO4": nk.BaSO4(lam),
        "gold": nk.gold(lam),
        "H2O": nk.H2O(lam),
    }

    fig, ax = plt.subplots(figsize=(7, 3))
    for label in ("CaCO3", "BaSO4", "gold"):
        ax.plot(lam, materials[label].real, label=f"{label}: n")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Real refractive index")
    ax.set_title("Offline material shortcut functions")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "materials": materials, "fig": fig}


def vo2_demo():
    lam = np.linspace(0.80, 5.00, 220)
    n_cold = nk.VO2(lam, T=25)
    n_hot = nk.VO2(lam, T=90)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, n_cold.real, label="VO2 at 25 C")
    ax.plot(lam, n_hot.real, label="VO2 at 90 C")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Real refractive index")
    ax.set_title("Temperature-driven VO2 response")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "n_cold": n_cold, "n_hot": n_hot, "fig": fig}


def run_all():
    return {
        "local_material": local_material_demo(),
        "oscillator": oscillator_demo(),
        "fit": fit_demo(),
        "effective_medium": effective_medium_demo(),
        "local_shortcuts": local_shortcuts_demo(),
        "vo2": vo2_demo(),
    }
