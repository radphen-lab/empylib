from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import IntegrationWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.examples import (
    color_system_examples,
    miescattering_examples,
    nklib_examples,
    rad_transfer_examples,
    ref_spectra_examples,
    scuffem_examples,
    spectrometry_examples,
    waveoptics_examples,
)
from docs.generate_tutorial_notebooks import NOTEBOOK_SPECS

import empylib.nklib as nk

ASSERT_ONLINE_CODECELLS = False

NOTEBOOK_REQUIRED_TOKENS = {
    "nklib_test.ipynb": [
        "nk.get_nkfile",
        "nk.blend_model",
        "nk.gaussian",
        "nk.tauc_lorentz",
        "nk.lorentz",
        "nk.drude",
        "nk.multi_oscillator",
        "nk.fit_to_oscillator",
        "nk.emt_multilayer_sphere",
        "nk.emt_brugg",
        "nk.eps_real_kkr",
        "nk.VO2",
        "nk.CaCO3",
        "nk.BaSO4",
        "nk.BiVO4_mono_a",
        "nk.BiVO4_mono_b",
        "nk.BiVO4_mono_c",
        "nk.BiVO4",
        "nk.Cu2O",
        "nk.MgO",
        "nk.GSTa",
        "nk.GSTc",
        "nk.VO2M",
        "nk.VO2R",
        "nk.gold",
        "nk.silver",
        "nk.Cu",
        "nk.Al",
        "nk.HDPE",
        "nk.PDMS",
        "nk.PVDF",
        "nk.H2O",
        "nk.ri_info_data",
        "nk.get_ri_info",
        "nk.SiO2",
        "nk.Silica",
        "nk.BaF2",
        "nk.TiO2",
        "nk.ZnO",
        "nk.Al2O3",
        "nk.ZnS",
        "nk.Si",
        "nk.Mg",
        "nk.PMMA",
    ],
    "ref_spectra_test.ipynb": [
        "ref.read_spectrafile",
        "ref.AM15",
        "ref.T_atmosphere",
        "ref.T_atmosphere_hemi",
        "ref.Bplanck",
        "ref.yCIE_lum",
        "ref.spectral_average",
        "ref.plot_spectra",
    ],
    "waveoptics_test.ipynb": [
        "wv.interface",
        "wv.multilayer",
        "wv.incoh_multilayer",
        "wv.snell",
    ],
    "miescattering_test.ipynb": [
        "mie.scatter_efficiency",
        "mie.scatter_coefficients",
        "mie.scatter_amplitude",
        "mie.scatter_stokes",
        "mie.phase_scatt_HG",
        "mie.scatter_from_phase_function",
        "mie.structure_factor_PY",
        "mie.phase_scatt_ensemble",
        "mie.cross_section_ensemble",
    ],
    "rad_transfer_test.ipynb": [
        "rt.T_beer_lambert",
        "rt.adm_sphere",
        "rt.adm",
    ],
    "spectrometry_test.ipynb": [
        "spec.read_uvvis",
        "spec.find_uvvis_samples",
        "spec.sample_uvvis",
    ],
    "scuffem_test.ipynb": [
        "scf.make_spectral_files",
        "scf.read_scatter_PFT",
        "scf.read_avescatter",
        "scf.clean_data",
    ],
    "color_system_test.ipynb": [
        "cs.spectrum_to_hex",
    ],
}

NKLIB_ONLINE_ONLY_TOKENS = [
    "nk.ri_info_data",
    "nk.get_ri_info",
    "nk.SiO2",
    "nk.Silica",
    "nk.BaF2",
    "nk.TiO2",
    "nk.ZnO",
    "nk.Al2O3",
    "nk.ZnS",
    "nk.Si",
    "nk.Mg",
    "nk.PMMA",
]

NKLIB_OFFLINE_SHORTCUTS = [
    "CaCO3",
    "BaSO4",
    "BiVO4_mono_a",
    "BiVO4_mono_b",
    "BiVO4_mono_c",
    "BiVO4",
    "Cu2O",
    "MgO",
    "GSTa",
    "GSTc",
    "VO2M",
    "VO2R",
    "gold",
    "silver",
    "Cu",
    "Al",
    "HDPE",
    "PDMS",
    "PVDF",
    "H2O",
]


def _token_pattern(token: str) -> re.Pattern[str]:
    return re.compile(rf"{re.escape(token)}(?![A-Za-z0-9_])")


def _source_has_token(source: str, token: str) -> bool:
    return _token_pattern(token).search(source) is not None


def _close_fig(payload):
    fig = payload.get("fig") if isinstance(payload, dict) else None
    if fig is not None:
        plt.close(fig)


def _assert_finite(name, values):
    arr = np.asarray(values)
    assert np.all(np.isfinite(arr)), f"{name} contains non-finite values"


def _assert_between(name, values, lower, upper):
    arr = np.asarray(values, dtype=float)
    assert np.all(arr >= lower), f"{name} has values below {lower}"
    assert np.all(arr <= upper), f"{name} has values above {upper}"


def _assert_commented_code_cell(cell):
    if cell.get("cell_type") != "code":
        return
    lines = "".join(cell.get("source", []))
    for line in lines.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            raise AssertionError("Optional online appendix code must stay commented out.")


def check_notebook_structure():
    docs_dir = ROOT / "docs"
    assert set(NOTEBOOK_REQUIRED_TOKENS) == set(NOTEBOOK_SPECS), "Notebook spec coverage drifted."

    for filename, required_tokens in NOTEBOOK_REQUIRED_TOKENS.items():
        path = docs_dir / filename
        assert path.is_file(), f"Missing notebook: {path}"

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert isinstance(payload.get("cells"), list) and payload["cells"], f"{filename} has no cells"

        sources = ["".join(cell.get("source", [])) for cell in payload["cells"]]
        full_source = "\n".join(sources)

        assert "docs.examples" not in full_source, f"{filename} still imports docs.examples"
        assert re.search(r"\bex\.", full_source) is None, f"{filename} still references helper wrappers"

        for token in required_tokens:
            assert _source_has_token(full_source, token), f"{filename} is missing required token {token}"

        if filename == "nklib_test.ipynb":
            appendix_idx = next(
                (i for i, src in enumerate(sources) if "Optional Appendix" in src),
                None,
            )
            assert appendix_idx is not None, "nklib notebook is missing the optional appendix heading"

            if ASSERT_ONLINE_CODECELLS:
                for token in NKLIB_ONLINE_ONLY_TOKENS:
                    token_cells = [i for i, src in enumerate(sources) if _source_has_token(src, token)]
                    assert token_cells, f"nklib notebook is missing {token}"
                    assert all(i > appendix_idx for i in token_cells), f"{token} must appear only in the optional appendix"
                    for idx in token_cells:
                        _assert_commented_code_cell(payload["cells"][idx])


def check_nklib():
    results = nklib_examples.run_all()
    _assert_finite("nklib local nk", results["local_material"]["nk"])
    assert not results["local_material"]["table"].empty
    _assert_finite("nklib oscillator", results["oscillator"]["nk"])
    assert results["fit"]["result"].success
    _assert_finite("nklib fitted nk", results["fit"]["fitted"])
    assert np.isfinite(results["effective_medium"]["n_eff_mix"])
    _assert_finite("nklib eps_real", results["effective_medium"]["eps_real"])
    assert set(results["local_shortcuts"]["materials"]) >= {"CaCO3", "BaSO4", "gold"}
    _assert_finite("nklib VO2 cold", results["vo2"]["n_cold"])
    _assert_finite("nklib VO2 hot", results["vo2"]["n_hot"])
    for payload in results.values():
        _close_fig(payload)

    lam = np.linspace(0.40, 2.00, 180)
    n_sio2, nk_table = nk.get_nkfile(lam, "sio2_Palik_Lemarchand2013", get_from_local_path=True)
    lower_edge = nk_table.index[int(0.25 * len(nk_table))]
    upper_edge = nk_table.index[int(0.75 * len(nk_table))]
    trusted = nk_table.loc[(nk_table.index >= lower_edge) & (nk_table.index <= upper_edge)]
    n_lorentz = nk.lorentz(lam, epsinf=1.8, wp=6.0, wn=7.5, gamma=0.8)
    n_blended = nk.blend_model(lam, trusted, n_lorentz, blend_low=0.08, blend_high=0.08)
    _assert_finite("nklib blended", n_blended)

    lam_model = np.linspace(0.45, 1.80, 80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        n_gaussian = nk.gaussian(lam_model, A=1.1, Br=0.30, E0=2.0)
        n_tauc = nk.tauc_lorentz(lam_model, A=10.0, C=1.2, E0=4.0, Eg=1.8)
    n_drude = nk.drude(lam_model, epsinf=1.0, wp=5.5, gamma=0.20)
    _assert_finite("nklib gaussian", n_gaussian)
    _assert_finite("nklib tauc_lorentz", n_tauc)
    _assert_finite("nklib drude", n_drude)

    lam_short = np.linspace(0.90, 1.10, 5)
    lam_gst = np.linspace(2.70, 3.00, 5)
    for name in NKLIB_OFFLINE_SHORTCUTS:
        func = getattr(nk, name)
        if name in {"VO2M", "VO2R"}:
            values = func(lam_short, film=2)
        elif name in {"GSTa", "GSTc"}:
            values = func(lam_gst)
        else:
            values = func(lam_short)
        _assert_finite(f"nklib shortcut {name}", values)


def check_ref_spectra():
    results = ref_spectra_examples.run_all()
    _assert_finite("ref spectra AM15 global", results["spectra"]["am15_global"])
    _assert_between("ref spectra atmosphere", results["spectra"]["t_atm"], 0.0, 1.0)
    _assert_between("ref spectra hemi atmosphere", results["spectra"]["t_atm_hemi"], 0.0, 1.0)
    _assert_finite("ref spectra Bplanck", results["spectra"]["b_300"])
    _assert_between("ref luminosity", results["luminosity"]["y_lum"], 0.0, 1.5)
    assert 0.0 <= results["average"]["solar_average"] <= 1.0
    assert 0.0 <= results["average"]["thermal_average"] <= 1.0
    for payload in results.values():
        _close_fig(payload)


def check_waveoptics():
    results = waveoptics_examples.run_all()
    _assert_between("waveoptics interface R", results["interface"]["R"], 0.0, 1.05)
    _assert_between("waveoptics interface T", results["interface"]["T"], 0.0, 2.0)
    _assert_between("waveoptics multilayer R", results["multilayer"]["R"], 0.0, 1.05)
    _assert_between("waveoptics multilayer T", results["multilayer"]["T"], 0.0, 2.0)
    _assert_between("waveoptics incoherent R", results["incoherent"]["R"], 0.0, 1.05)
    _assert_between("waveoptics incoherent T", results["incoherent"]["T"], 0.0, 2.0)
    assert math.isfinite(float(results["snell"]["theta_t_deg"]))
    for payload in results.values():
        _close_fig(payload)


def check_miescattering():
    results = miescattering_examples.run_all()
    _assert_between("miescattering g", results["single_sphere"]["gcos"], -1.0, 1.0)
    _assert_between("miescattering coated g", results["coated_sphere"]["gcos"], -1.0, 1.0)
    assert results["coated_sphere"]["an"].shape[0] == results["coated_sphere"]["lam"].size
    assert results["angular"]["s11"].shape[0] == results["angular"]["theta"].size
    _assert_between("miescattering qsca_hg", results["ensemble"]["qsca_hg"], 0.0, 10.0)
    _assert_between("miescattering g_hg", results["ensemble"]["g_hg"], -1.0, 1.0)
    _assert_finite("miescattering structure factor", results["ensemble"]["structure_factor"])
    _assert_between("miescattering ensemble g", results["ensemble"]["g_av"], -1.0, 1.0)
    assert isinstance(results["ensemble"]["phase_df"], pd.DataFrame)
    for payload in results.values():
        _close_fig(payload)


def check_rad_transfer():
    results = rad_transfer_examples.run_all()
    beer_cols = ["Rtot", "Ttot", "Tspec", "Tdif"]
    adm_cols = ["Rtot", "Ttot", "Rspec", "Tspec", "Rdif", "Tdif"]

    df = results["beer_lambert"]["result"]
    assert list(df.columns) == beer_cols
    assert isinstance(df.index.name, str) and df.index.name.startswith("Wavelength")
    _assert_between("beer_lambert flux", df.values, 0.0, 1.05)

    df = results["adm_sphere"]["result"]
    assert list(df.columns) == adm_cols
    assert isinstance(df.index.name, str) and df.index.name.startswith("Wavelength")
    _assert_between("adm_sphere flux", df.values, 0.0, 1.05)

    for key in ("result_g", "result_pf"):
        df = results["adm_comparison"][key]
        assert list(df.columns) == adm_cols
        assert isinstance(df.index.name, str) and df.index.name.startswith("Wavelength")
        _assert_between(f"adm comparison {key}", df.values, 0.0, 1.05)
    for payload in results.values():
        _close_fig(payload)


def check_spectrometry():
    results = spectrometry_examples.run_all()
    assert isinstance(results["read_uvvis"]["dataset"], pd.DataFrame)
    assert not results["read_uvvis"]["dataset"].empty
    assert len(results["find_samples"]["sample_names"]) >= 1
    dataset = results["sample"]["dataset"]
    assert isinstance(dataset, pd.DataFrame)
    assert not dataset.empty
    assert any(col.startswith("R") or col.startswith("T") for col in dataset.columns)


def check_scuffem():
    results = scuffem_examples.run_all()
    assert len(results["make_files"]["omega_lines"]) == results["make_files"]["lam"].size
    assert len(results["make_files"]["material_lines"]) == results["make_files"]["lam"].size + 2
    assert results["read_results"]["pft"]
    assert results["read_results"]["ave"]
    assert results["clean_data"]["cleaned"]


def check_color_system():
    results = color_system_examples.run_all()
    for payload in results.values():
        assert payload["hex_color"].startswith("#") and len(payload["hex_color"]) == 7
        _assert_between("color rgb01", payload["rgb01"], 0.0, 1.0)
        _assert_between("color rgb255", payload["rgb255"], 0.0, 255.0)


def main():
    check_notebook_structure()
    check_nklib()
    check_ref_spectra()
    check_waveoptics()
    check_miescattering()
    check_rad_transfer()
    check_spectrometry()
    check_scuffem()
    check_color_system()
    print("All tutorial examples passed.")


if __name__ == "__main__":
    main()
