from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import empylib.miescattering as mie


def single_sphere_demo():
    lam = np.linspace(0.40, 1.00, 121)
    qabs, qsca, gcos = mie.scatter_efficiency(
        lam,
        1.50,
        2.35 + 0.01j + 0 * lam,
        0.30,
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, qabs, label="Qabs")
    ax.plot(lam, qsca, label="Qsca")
    ax.plot(lam, gcos, label="g")
    ax.set_xlabel("Wavelength (um)")
    ax.set_title("Single-sphere scattering efficiencies")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "qabs": qabs, "qsca": qsca, "gcos": gcos, "fig": fig}


def coated_sphere_demo():
    lam = np.linspace(0.40, 1.00, 121)
    np_shells = [1.80 + 0.00j + 0 * lam, 2.35 + 0.01j + 0 * lam]
    d_shells = np.array([0.18, 0.30])
    qabs, qsca, gcos = mie.scatter_efficiency(lam, 1.50, np_shells, d_shells)
    an, bn = mie.scatter_coefficients(lam, 1.50, np_shells, d_shells)

    return {
        "lam": lam,
        "qabs": qabs,
        "qsca": qsca,
        "gcos": gcos,
        "an": an,
        "bn": bn,
    }


def angular_demo():
    lam = np.linspace(0.40, 1.00, 121)
    theta = np.linspace(0.0, np.pi, 181)
    s1, s2 = mie.scatter_amplitude(lam, 1.50, 2.35 + 0.01j + 0 * lam, 0.30, theta=theta)
    s11, s12, s33, s34 = mie.scatter_stokes(lam, 1.50, 2.35 + 0.01j + 0 * lam, 0.30, theta=theta)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.degrees(theta), s11[:, 20], label="S11 at one wavelength")
    ax.plot(np.degrees(theta), s12[:, 20], label="S12 at one wavelength")
    ax.set_xlabel("Scattering angle (deg)")
    ax.set_title("Angular scattering response")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {
        "lam": lam,
        "theta": theta,
        "s1": s1,
        "s2": s2,
        "s11": s11,
        "s12": s12,
        "s33": s33,
        "s34": s34,
        "fig": fig,
    }


def ensemble_demo():
    lam = np.linspace(0.40, 1.00, 121)
    theta = np.linspace(0.0, np.pi, 181)
    d_bins = np.linspace(0.20, 0.40, 7)
    size_dist = np.exp(-((d_bins - 0.30) / 0.05) ** 2)
    size_dist /= size_dist.sum()

    hg_phase = mie.phase_scatt_HG(
        lam,
        np.full(lam.size, 0.7),
        qsca=np.ones_like(lam),
        theta=theta,
    )
    qsca_hg, g_hg = mie.scatter_from_phase_function(hg_phase)
    structure_factor = mie.structure_factor_PY(
        lam,
        1.50,
        d_bins,
        fv=0.08,
        theta=theta,
        size_dist=size_dist,
    )
    phase_ensemble = mie.phase_scatt_ensemble(
        lam,
        1.50,
        2.35 + 0.01j + 0 * lam,
        d_bins,
        fv=0.08,
        size_dist=size_dist,
        theta=theta,
    )
    cabs, csca, g_av, phase_df = mie.cross_section_ensemble(
        lam,
        1.50,
        2.35 + 0.01j + 0 * lam,
        d_bins,
        fv=0.08,
        size_dist=size_dist,
        theta=theta,
        phase_function=True,
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, cabs, label="Cabs")
    ax.plot(lam, csca, label="Csca")
    ax.plot(lam, g_av, label="g")
    ax.set_xlabel("Wavelength (um)")
    ax.set_title("Ensemble-averaged optical response")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {
        "lam": lam,
        "theta": theta,
        "d_bins": d_bins,
        "size_dist": size_dist,
        "hg_phase": hg_phase,
        "qsca_hg": qsca_hg,
        "g_hg": g_hg,
        "structure_factor": structure_factor,
        "phase_ensemble": phase_ensemble,
        "cabs": cabs,
        "csca": csca,
        "g_av": g_av,
        "phase_df": phase_df,
        "fig": fig,
    }


def run_all():
    return {
        "single_sphere": single_sphere_demo(),
        "coated_sphere": coated_sphere_demo(),
        "angular": angular_demo(),
        "ensemble": ensemble_demo(),
    }
