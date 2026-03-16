from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import empylib.waveoptics as wv


def interface_demo():
    lam = np.linspace(0.40, 1.00, 160)
    aoi = np.radians([0.0, 30.0, 60.0])
    n_below = 1.50 + 0.01j + 0 * lam
    R, T, r, t = wv.interface(1.00, n_below, aoi=aoi, polarization="TM")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, R[0], label="R at 0 deg")
    ax.plot(lam, R[1], label="R at 30 deg")
    ax.plot(lam, R[2], label="R at 60 deg")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Single-interface Fresnel response")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "aoi": aoi, "R": R, "T": T, "r": r, "t": t, "fig": fig}


def multilayer_demo():
    lam = np.linspace(0.40, 1.00, 180)
    n_layers = [
        1.45 + 0.00j + 0 * lam,
        2.10 + 0.02j + 0 * lam,
    ]
    thickness = [0.12, 0.05]
    R, T, r, t = wv.multilayer(
        lam,
        aoi=np.radians(15.0),
        N_layers=n_layers,
        thickness=thickness,
        N_above=1.00,
        N_below=1.52,
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, R, label="R")
    ax.plot(lam, T, label="T")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Flux")
    ax.set_title("Coherent multilayer film")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "R": R, "T": T, "r": r, "t": t, "fig": fig}


def incoherent_demo():
    lam = np.linspace(0.40, 1.00, 180)
    n_layers = [
        1.45 + 0.00j + 0 * lam,
        2.10 + 0.02j + 0 * lam,
    ]
    thickness = [0.12, 0.05]
    R, T = wv.incoh_multilayer(
        lam,
        N_layers=n_layers,
        thickness=thickness,
        aoi=np.radians(15.0),
        N_above=1.00,
        N_below=1.52,
        coh_length=0.03,
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lam, R, label="R")
    ax.plot(lam, T, label="T")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Flux")
    ax.set_title("Partially incoherent multilayer film")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return {"lam": lam, "R": R, "T": T, "fig": fig}


def snell_demo():
    theta_i = np.radians(35.0)
    theta_t = wv.snell(1.00, 1.50, theta_i)
    return {
        "theta_i_deg": np.degrees(theta_i),
        "theta_t_deg": np.degrees(theta_t),
    }


def run_all():
    return {
        "interface": interface_demo(),
        "multilayer": multilayer_demo(),
        "incoherent": incoherent_demo(),
        "snell": snell_demo(),
    }
