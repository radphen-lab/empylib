from __future__ import annotations

import numpy as np

from .bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import empylib.color_system as cs
import empylib.ref_spectra as ref


def material_color_demo():
    lam = np.linspace(0.38, 0.78, 201)
    reflectance = 0.15 + 0.75 * np.exp(-((lam - 0.62) / 0.08) ** 2)
    hex_color, rgb01, rgb255 = cs.spectrum_to_hex(
        lam,
        reflectance,
        source="material",
        illuminant_name="D65",
    )
    return {
        "lam": lam,
        "reflectance": reflectance,
        "hex_color": hex_color,
        "rgb01": rgb01,
        "rgb255": rgb255,
    }


def emitter_color_demo():
    lam = np.linspace(0.38, 0.78, 201)
    emitter_spd = ref.Bplanck(lam, T=2600, unit="wavelength")
    hex_color, rgb01, rgb255 = cs.spectrum_to_hex(
        lam,
        emitter_spd,
        source="emitter",
        emitter_units="per_um",
    )
    return {
        "lam": lam,
        "emitter_spd": emitter_spd,
        "hex_color": hex_color,
        "rgb01": rgb01,
        "rgb255": rgb255,
    }


def custom_illuminant_demo():
    lam = np.linspace(0.38, 0.78, 201)
    reflectance = 0.20 + 0.70 * np.exp(-((lam - 0.55) / 0.10) ** 2)
    solar = ref.AM15(lam, spectra_type="global")
    hex_color, rgb01, rgb255 = cs.spectrum_to_hex(
        lam,
        reflectance,
        source="material",
        illuminant=(lam, solar),
        illuminant_units="per_um",
    )
    return {
        "lam": lam,
        "reflectance": reflectance,
        "solar": solar,
        "hex_color": hex_color,
        "rgb01": rgb01,
        "rgb255": rgb255,
    }


def run_all():
    return {
        "material": material_color_demo(),
        "emitter": emitter_color_demo(),
        "custom_illuminant": custom_illuminant_demo(),
    }
