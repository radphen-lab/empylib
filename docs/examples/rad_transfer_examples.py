from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import empylib.miescattering as mie
import empylib.rad_transfer as rt
from empylib.utils import rt_style_mapper


def beer_lambert_demo():
    lam = np.linspace(0.40, 1.00, 121)
    nh = 1.49 + 0.00j + 0 * lam
    npart = 2.35 + 0.02j + 0 * lam
    result = rt.T_beer_lambert(
        wavelength = lam,
        N_host = nh,
        N_particle = npart,
        D = 0.30,
        fv=0.06,
        thickness=0.50,
        aoi=np.radians(8.0),
        N_above=1.00,
        N_below=1.52,
    )

    styles, labels = rt_style_mapper(result)
    fig, ax = plt.subplots(figsize=(7, 3))
    for col in result.columns:
        ax.plot(result.index, result[col], styles[col], label=labels.get(col, col))
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Flux")
    ax.set_title("Beer-Lambert slab with embedded particles")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "result": result, "fig": fig}


def adm_sphere_demo():
    lam = np.linspace(0.40, 1.00, 121)
    nh = 1.49 + 0.00j + 0 * lam
    npart = 2.35 + 0.02j + 0 * lam
    result = rt.adm_sphere(
        lam,
        nh,
        npart,
        0.30,
        fv=0.06,
        thickness=0.50,
        N_above=1.00,
        N_below=1.52,
        dependent_scatt=False,
        effective_medium=True,
        use_phase_fun=False,
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    for col in result.columns:
        ax.plot(result.index, result[col], label=col)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Flux")
    ax.set_title("Adding-doubling model with spherical particles")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "result": result, "fig": fig}


def adm_comparison_demo():
    lam = np.linspace(0.40, 1.00, 121)
    nh = 1.49 + 0.00j + 0 * lam
    k_sca = 0.8 * np.exp(-((lam - 0.65) / 0.16) ** 2)
    k_abs = 0.2 * np.exp(-((lam - 0.82) / 0.12) ** 2)
    gcos = np.clip(0.2 + 0.6 * (lam - lam.min()) / (lam.max() - lam.min()), 0, 0.95)

    result_g = rt.adm(
        lam,
        thickness=0.30,
        k_sca=k_sca,
        k_abs=k_abs,
        N_host=nh,
        gcos=gcos,
        N_above=1.00,
        N_below=1.52,
    )

    phase_fun = mie.phase_scatt_HG(
        lam,
        gcos=np.full(lam.size, 0.75),
        qsca=np.ones_like(lam),
        theta=np.linspace(0.0, np.pi, 181),
    )
    result_pf = rt.adm(
        lam,
        thickness=0.30,
        k_sca=k_sca,
        k_abs=k_abs,
        N_host=nh,
        phase_fun=phase_fun,
        N_above=1.00,
        N_below=1.52,
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, result_g["Ttot"], label="Ttot from g")
    ax.plot(lam, result_pf["Ttot"], label="Ttot from phase function")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Total transmission")
    ax.set_title("ADM with g-only vs full phase function")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {
        "lam": lam,
        "result_g": result_g,
        "result_pf": result_pf,
        "phase_fun": phase_fun,
        "fig": fig,
    }


def run_all():
    return {
        "beer_lambert": beer_lambert_demo(),
        "adm_sphere": adm_sphere_demo(),
        "adm_comparison": adm_comparison_demo(),
    }
