from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import empylib.ref_spectra as ref


def spectra_demo():
    lam = np.linspace(0.30, 25.0, 400)
    atm_values, atm_table = ref.read_spectrafile(
        lam,
        "T_atmosphere.txt",
        get_from_local_path=True,
        return_data=True,
    )
    am15_global = ref.AM15(lam, spectra_type="global")
    am15_direct = ref.AM15(lam, spectra_type="direct")
    t_atm = ref.T_atmosphere(lam)
    t_atm_hemi = ref.T_atmosphere_hemi(lam, beta_tilt=30)
    b_300 = ref.Bplanck(lam, T=300, unit="wavelength")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lam, am15_global / am15_global.max(), label="AM1.5 global")
    ax.plot(lam, am15_direct / am15_direct.max(), label="AM1.5 direct")
    ax.plot(lam, t_atm, label="Atmospheric transmission")
    ax.plot(lam, t_atm_hemi, label="Hemispherical atmosphere")
    ax.plot(lam, b_300 / np.nanmax(b_300), label="Planck at 300 K")
    ax.set_xlabel("Wavelength (um)")
    ax.set_title("Reference spectra used across EMPI Lib")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {
        "lam": lam,
        "atm_values": atm_values,
        "atm_table": atm_table,
        "am15_global": am15_global,
        "am15_direct": am15_direct,
        "t_atm": t_atm,
        "t_atm_hemi": t_atm_hemi,
        "b_300": b_300,
        "fig": fig,
    }


def luminosity_demo():
    lam = np.linspace(0.38, 0.78, 200)
    y_lum = ref.yCIE_lum(lam)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, y_lum, color="tab:green")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Relative response")
    ax.set_title("Photopic luminosity function")
    ax.grid(True, alpha=0.3)

    return {"lam": lam, "y_lum": y_lum, "fig": fig}


def average_demo():
    lam = np.linspace(0.30, 25.0, 400)
    absorptance = 0.15 + 0.75 * np.exp(-((lam - 1.10) / 0.60) ** 2)
    solar_average = ref.spectral_average(lam, absorptance, spectrum="solar")
    thermal_average = ref.spectral_average(lam, absorptance, spectrum="thermal", T=500)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, absorptance, color="black", label="Absorptance")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Spectral value")
    ax.set_title("Example property for spectral averaging")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {
        "lam": lam,
        "absorptance": absorptance,
        "solar_average": solar_average,
        "thermal_average": thermal_average,
        "fig": fig,
    }


def plotting_demo():
    lam = np.linspace(0.30, 25.0, 400)
    absorptance = 0.15 + 0.75 * np.exp(-((lam - 1.10) / 0.60) ** 2)
    fig, ax = ref.plot_spectra(
        (lam, absorptance, {"label": "Absorptance", "color": "black"}),
        (lam, 1 - absorptance, {"label": "Reflectance surrogate", "color": "tab:blue"}),
        ylabel="Dimensionless spectrum",
        title="Using plot_spectra with standard backgrounds",
        show_background_legend=True,
    )
    return {"lam": lam, "absorptance": absorptance, "fig": fig, "ax": ax}


def run_all():
    return {
        "spectra": spectra_demo(),
        "luminosity": luminosity_demo(),
        "average": average_demo(),
        "plotting": plotting_demo(),
    }
