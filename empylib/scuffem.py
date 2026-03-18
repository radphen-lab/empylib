# -*- coding: utf-8 -*-
"""
Library of functions for scuffem

Created on Thu Oct 10 14:12 2024

@author: PanxoPanza
"""

import copy as _copy

import numpy as np
import pandas as pd

from .utils import detect_spectral_spikes as _detect_spectral_spikes


def make_spectral_files(wavelength, Material=None):
    """
    Create OmegaList and dielectric properties files for scuff-EM simulations.

    Input:
        wavelength: Wavelength range (um)
        Material: dictionary with:
            keys: materials name for .dat file
            values: nk data in ndarray (dtype=complex)
    """
    c0 = 299792458  # speed of light in m/s

    if Material is None:
        Material = {}
    elif not isinstance(Material, dict):
        raise ValueError("Material variable must be a dictionary")

    with open("OmegaList.dat", "w") as f:
        for iw in range(len(wavelength)):
            f.write(f"{2 * np.pi / wavelength[iw]:.6f}")
            if iw < len(wavelength) - 1:
                f.write("\n")

    lambda_mat = np.insert(wavelength, 0, min(wavelength) * 0.9)
    lambda_mat = np.append(lambda_mat, max(wavelength) * 1.1)
    w = 2 * np.pi * c0 / lambda_mat * 1e6

    for mat_label, nk_raw in Material.items():
        if not isinstance(nk_raw, np.ndarray):
            raise ValueError(f'"{mat_label}" values are not ndarray')
        if len(wavelength) != len(nk_raw):
            raise ValueError(f'size of "{mat_label}" and "wavelength" arrays must be equal')

        nk_data = np.interp(lambda_mat, wavelength, nk_raw.astype(complex))
        eps = nk_data**2

        with open(f"{mat_label}.dat", "w") as f:
            for wi, epsilon in zip(w, eps):
                f.write(f"{wi:.6e} {epsilon.real:.5e}+{epsilon.imag:.5e}i\n")


def _group_scuff_table(df, drop_columns):
    df = df.copy()
    df.set_index("Omega", inplace=True)
    df.index.name = "Omega"

    objectID = {}
    for label, group in df.groupby("Label"):
        objectID[label] = group.drop(columns=list(drop_columns), errors="ignore")

    return objectID


def read_scatter_PFT(FileName):
    """
    Read scuff-EM scattering output into a dictionary of DataFrames keyed by label.

    The parser accepts both the detailed 10-column PFT table and the shorter
    6-column average-scattering layout shipped with the current docs bundle.
    """
    df = pd.read_csv(FileName, comment="#", sep=r"\s+", header=None)
    if df.empty:
        return {}

    if df.shape[1] == 10:
        df.columns = ["Omega", "Label", "Pabs", "Psca", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        return _group_scuff_table(df, drop_columns=("Label",))

    if df.shape[1] == 6:
        df.columns = ["Transform", "Omega", "Label", "Pabs", "Psca", "gPsca"]
        return _group_scuff_table(df, drop_columns=("Label", "Transform"))

    raise ValueError(
        f"Unsupported scatter PFT format in '{FileName}': expected 6 or 10 columns, got {df.shape[1]}."
    )


def read_avescatter(FileName):
    """
    Read orientation-averaged scattering output into a dictionary of DataFrames keyed by label.
    """
    df = pd.read_csv(FileName, comment="#", sep=r"\s+", header=None)
    if df.empty:
        return {}

    if df.shape[1] == 6:
        df.columns = ["Transform", "Omega", "Label", "<Cabs>", "<Csca>", "<Cpr>"]
        return _group_scuff_table(df, drop_columns=("Label", "Transform"))

    if df.shape[1] == 5:
        df.columns = ["Omega", "Label", "<Cabs>", "<Csca>", "<Cpr>"]
        return _group_scuff_table(df, drop_columns=("Label",))

    raise ValueError(
        f"Unsupported average-scatter format in '{FileName}': expected 5 or 6 columns, got {df.shape[1]}."
    )


def clean_data(
    objectID,
    k: float = 4.0,
    min_slope: float | None = None,
    dilate: int = 0,
    max_frac_removed: float = 0.25,
    inplace=True,
):
    if inplace:
        object_fix = objectID
    else:
        object_fix = _copy.deepcopy(objectID)

    for df in object_fix.values():
        if not isinstance(df, pd.DataFrame):
            continue

        for col in df.keys():
            y = df[col].values
            x = df.index.values

            y_clean, mask = _detect_spectral_spikes(
                x,
                y,
                k=k,
                min_slope=min_slope,
                dilate=dilate,
                max_frac_removed=max_frac_removed,
                return_mask=True,
            )
            if len(mask) == 0:
                continue

            df[col] = y_clean

    if not inplace:
        return object_fix

    return None
