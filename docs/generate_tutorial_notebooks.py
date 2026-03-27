from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def bullet_lines(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def setup_cell(import_block: str) -> dict:
    return code(
        f"""
        from pathlib import Path
        import os
        import sys

        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            if (candidate / "empylib").exists() and (candidate / "docs").exists():
                ROOT = candidate
                break
        else:
            raise FileNotFoundError("Could not locate the EMPI Lib repository root.")

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from IPython.display import display
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        plt.rcParams["figure.figsize"] = (7, 3)

        {dedent(import_block).strip()}
        """
    )


def intro_cell(title: str, summary: str, goals: list[str]) -> dict:
    return md(
        "\n".join(
            [
                f"# `{title}` tutorial",
                "",
                summary,
                "",
                "**Learning goals**",
                "",
                bullet_lines(goals),
                "",
                "**Notebook design**",
                "",
                "- every runnable cell calls the public `empylib` API directly",
                "- parameter meanings are explained in markdown and in short inline comments",
                "- outputs are inspected in the same notebook so you can see what each function returns",
                "- the core path is offline-first; internet-backed examples live in clearly marked optional appendices",
            ]
        )
    )


def section_cells(spec: dict) -> list[dict]:
    return [
        md(
            "\n".join(
                [
                    f"## {spec['title']}",
                    "",
                    "**Functions used**",
                    "",
                    bullet_lines(spec["functions"]),
                    "",
                    "**Problem we are solving**",
                    "",
                    spec["problem"],
                    "",
                    "**Parameter guide for this example**",
                    "",
                    bullet_lines(spec["parameters"]),
                    "",
                    "**Outputs to inspect**",
                    "",
                    bullet_lines(spec["outputs"]),
                ]
            )
        ),
        code(spec["code"]),
        md(
            "\n".join(
                [
                    "**How to read the result**",
                    "",
                    spec["read"],
                    "",
                    "**Common pitfalls**",
                    "",
                    bullet_lines(spec["pitfalls"]),
                    "",
                    "**Try this next**",
                    "",
                    bullet_lines(spec["next"]),
                ]
            )
        ),
    ]


def appendix_cells(title: str, explanation: str, code_source: str) -> list[dict]:
    return [
        md(
            "\n".join(
                [
                    f"## Optional Appendix: {title}",
                    "",
                    explanation,
                ]
            )
        ),
        code(code_source),
    ]


NOTEBOOK_SPECS = {}


NOTEBOOK_SPECS["nklib_test.ipynb"] = {
    "title": "nklib",
    "summary": "This notebook teaches the refractive-index tools in `empylib.nklib`: tabulated files, oscillator models, effective-medium approximations, and material shortcuts. The core examples are fully offline and use the packaged `.nk` files shipped with the repository.",
    "goals": [
        "load local optical-constant tables with `get_nkfile`",
        "understand when to use tabulated data, oscillator models, and blended extrapolation",
        "fit a synthetic spectrum with `fit_to_oscillator` and inspect the fitted parameters",
        "use EMT and VO2 utilities, then survey the offline material shortcut catalog",
    ],
    "setup": """
    import warnings

    from scipy.integrate import IntegrationWarning

    import empylib.nklib as nk
    """,
    "sections": [
        {
            "title": "Load a local `.nk` file and inspect the interpolated output",
            "functions": ["nk.get_nkfile"],
            "problem": "A common starting point is a tabulated material file. Here we load silica from the packaged database, interpolate it on a wavelength grid, and compare the interpolated `n + ik` output against the raw table.",
            "parameters": [
                "`wavelength`: wavelength grid in micrometers where you want the interpolated refractive index",
                "`MaterialName`: base name of the `.nk` file without the extension",
                "`get_from_local_path=True`: search inside `empylib/nk_files` instead of the current working directory",
                "`lam_units='um'`: declares the units of the wavelength grid you passed in",
            ],
            "outputs": [
                "`n_sio2`: complex array `n + ik` evaluated on `wavelength`",
                "`nk_table`: original tabulated data as a DataFrame indexed by wavelength",
                "a quick plot showing the interpolated real and imaginary parts",
            ],
            "code": """
            lam = np.linspace(0.40, 2.00, 250)

            n_sio2, nk_table = nk.get_nkfile(
                lam,
                "sio2_Palik_Lemarchand2013",  # packaged file name without ".nk"
                get_from_local_path=True,      # read from empylib/nk_files
                lam_units="um",                # lam is already in micrometers
            )

            display(nk_table.head())
            print("Interpolated array shape:", n_sio2.shape)
            print("First complex value:", n_sio2[0])

            fig, ax = plt.subplots()
            ax.plot(lam, n_sio2.real, label="n")
            ax.plot(lam, n_sio2.imag, label="k")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Optical constants")
            ax.set_title("Silica loaded from a local .nk file")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The returned complex array always matches the wavelength grid you requested. The DataFrame is useful when you want to inspect the source table, its valid wavelength range, or reuse it for a custom interpolation workflow.",
            "pitfalls": [
                "Do not include the `.nk` extension in `MaterialName`",
                "If you set `get_from_local_path=False`, the function looks in the current working directory instead of the packaged database",
                "If you pass wavelengths outside the tabulated range, you should think explicitly about extrapolation behavior",
            ],
            "next": [
                "Try a different packaged material such as `CaCO3_Palik` or `GSTa_Du2016`",
                "Pass wavelengths in nanometers and change `lam_units` to `'nm'`",
            ],
        },
        {
            "title": "Blend tabulated data with a model outside a trusted range",
            "functions": ["nk.lorentz", "nk.blend_model"],
            "problem": "Sometimes you trust tabulated data only in a limited interval and want a smooth handoff to a model outside that interval. This example uses a Lorentz model as the outer behavior and the measured data in the middle.",
            "parameters": [
                "`nk_df`: DataFrame with `n` and `k` columns indexed by wavelength",
                "`nk_model`: model-predicted complex index on the same wavelength grid as `wavelength`",
                "`blend_low` and `blend_high`: smoothing windows near the lower and upper table edges",
                "`epsinf`, `wp`, `wn`, `gamma`: Lorentz parameters controlling high-frequency dielectric constant, strength, resonance, and damping",
            ],
            "outputs": [
                "`lorentz_model`: a physically motivated model for the full wavelength span",
                "`blended_nk`: tabulated values inside the trusted range and model values outside it",
                "a plot comparing the pure model and the blended result",
            ],
            "code": """
            lower_edge = nk_table.index[int(0.25 * len(nk_table))]
            upper_edge = nk_table.index[int(0.75 * len(nk_table))]
            trusted_table = nk_table.loc[(nk_table.index >= lower_edge) & (nk_table.index <= upper_edge)]

            lorentz_model = nk.lorentz(
                lam,
                epsinf=1.8,  # dielectric constant at very high energy
                wp=6.0,      # oscillator strength / plasma frequency in eV
                wn=7.5,      # resonance energy in eV
                gamma=0.8,   # damping in eV
            )

            blended_nk = nk.blend_model(
                lam,
                trusted_table,   # trusted n,k table in the middle of the spectrum
                lorentz_model,   # model used outside the tabulated interval
                blend_low=0.08,  # low-end smoothing window in micrometers
                blend_high=0.08, # high-end smoothing window in micrometers
            )

            fig, ax = plt.subplots()
            ax.plot(lam, lorentz_model.real, "--", label="Lorentz model: n")
            ax.plot(lam, blended_nk.real, label="Blended result: n")
            ax.axvspan(trusted_table.index.min(), trusted_table.index.max(), color="tab:green", alpha=0.10)
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Real refractive index")
            ax.set_title("Smooth transition between model and tabulated data")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "Inside the shaded region the result follows the table, while outside it follows the Lorentz model. The blend windows prevent a visible discontinuity at the handoff points.",
            "pitfalls": [
                "The DataFrame passed to `blend_model` must have `n` and `k` columns and a wavelength index",
                "Choose a model that is at least qualitatively compatible with the material outside the measured range",
                "If your blend windows are too wide, the model can distort part of the range you actually trust",
            ],
            "next": [
                "Swap the Lorentz model for a `multi_oscillator` model",
                "Reduce `blend_low` and `blend_high` to see where the transition becomes abrupt",
            ],
        },
        {
            "title": "Compare the individual oscillator families",
            "functions": ["nk.gaussian", "nk.tauc_lorentz", "nk.lorentz", "nk.drude"],
            "problem": "The oscillator utilities provide compact analytic models for different physical situations: bounded resonances, band-edge absorption, dielectric resonances, and free-carrier behavior. This section shows how each one is called.",
            "parameters": [
                "`A`, `Br`, `E0`: Gaussian amplitude, broadening, and center energy",
                "`A`, `C`, `E0`, `Eg`: Tauc-Lorentz amplitude, broadening, resonance energy, and bandgap",
                "`epsinf`, `wp`, `wn`, `gamma`: Lorentz dielectric background, strength, resonance, and damping",
                "`epsinf`, `wp`, `gamma`: Drude background, plasma frequency, and damping",
            ],
            "outputs": [
                "four complex spectra that you can compare directly",
                "a plot of the real refractive index from each oscillator family",
            ],
            "code": """
            lam_model = np.linspace(0.45, 1.80, 220)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IntegrationWarning)

                n_gaussian = nk.gaussian(
                    lam_model,
                    A=1.1,   # absorption amplitude
                    Br=0.30, # Gaussian broadening in eV
                    E0=2.0,  # center energy in eV
                )

                n_tauc = nk.tauc_lorentz(
                    lam_model,
                    A=10.0,  # oscillator amplitude
                    C=1.2,   # broadening in eV
                    E0=4.0,  # resonance energy in eV
                    Eg=1.8,  # bandgap in eV
                )

            n_lorentz = nk.lorentz(
                lam_model,
                epsinf=1.0,
                wp=7.5,
                wn=3.0,
                gamma=0.25,
            )

            n_drude = nk.drude(
                lam_model,
                epsinf=1.0,
                wp=5.5,
                gamma=0.20,
            )

            oscillator_preview = pd.DataFrame(
                {
                    "gaussian_n": n_gaussian.real,
                    "tauc_lorentz_n": n_tauc.real,
                    "lorentz_n": n_lorentz.real,
                    "drude_n": n_drude.real,
                },
                index=lam_model,
            )
            oscillator_preview.index.name = "Wavelength (um)"
            display(oscillator_preview.head())

            fig, ax = plt.subplots()
            ax.plot(lam_model, n_gaussian.real, label="gaussian")
            ax.plot(lam_model, n_tauc.real, label="tauc_lorentz")
            ax.plot(lam_model, n_lorentz.real, label="lorentz")
            ax.plot(lam_model, n_drude.real, label="drude")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Real refractive index")
            ax.set_title("Calling each oscillator family directly")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The shapes are intentionally different. Gaussian and Tauc-Lorentz are useful for bounded absorption features, while Lorentz and Drude are the classic dielectric and free-carrier building blocks.",
            "pitfalls": [
                "Oscillator parameters are in electron-volts, not micrometers",
                "The Gaussian and Tauc-Lorentz models reconstruct the real part through a Kramers-Kronig step, so extreme parameters can become numerically stiff",
                "A good-looking `n` curve does not guarantee the `k` curve is physically meaningful; inspect both when fitting real data",
            ],
            "next": [
                "Plot the imaginary parts as well to see where absorption is concentrated",
                "Increase `gamma` in the Lorentz or Drude model to broaden the response",
            ],
        },
        {
            "title": "Combine oscillators and fit them to a target spectrum",
            "functions": ["nk.multi_oscillator", "nk.fit_to_oscillator"],
            "problem": "Real materials often need more than one oscillator term. Here we build a synthetic target from a known oscillator dictionary, then recover the parameters with `fit_to_oscillator`.",
            "parameters": [
                "`oscillator_dict`: nested dictionary whose keys are oscillator family names",
                "`n_data` and `k_data`: measured or synthetic real and imaginary parts to fit",
                "`bounds`: optional parameter limits; omitted here to use the defaults",
                "`x_units='um'`: declares the wavelength units used in the fit input",
            ],
            "outputs": [
                "`target_nk`: synthetic target spectrum",
                "`fitted_oscillator`: dictionary with recovered parameters",
                "`result`: SciPy optimization result with convergence information",
            ],
            "code": """
            lam_fit = np.linspace(0.45, 1.80, 120)

            target_oscillator = {
                "lorentz": {
                    "epsinf": 1.0,
                    "wp": 7.0,
                    "wn": 2.8,
                    "gamma": 0.22,
                },
                "gaussian": {
                    "A": 0.4,
                    "Br": 0.18,
                    "E0": 2.3,
                },
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IntegrationWarning)
                target_nk = nk.multi_oscillator(lam_fit, target_oscillator)

            initial_guess = {
                "lorentz": {
                    "epsinf": 1.2,
                    "wp": 6.5,
                    "wn": 2.7,
                    "gamma": 0.30,
                },
                "gaussian": {
                    "A": 0.3,
                    "Br": 0.25,
                    "E0": 2.1,
                },
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IntegrationWarning)
                fitted_oscillator, result = nk.fit_to_oscillator(
                    lam_fit,
                    [target_nk.real, target_nk.imag],  # [n_data, k_data]
                    initial_guess,    # initial parameter dictionary
                    x_units="um",     # wavelength unit used by lam_fit
                )
                fitted_nk = nk.multi_oscillator(lam_fit, fitted_oscillator)

            print("Optimization succeeded:", result.success)
            print("Recovered oscillator dictionary:")
            display(pd.Series({key: str(value) for key, value in fitted_oscillator.items()}))

            fig, ax = plt.subplots()
            ax.plot(lam_fit, target_nk.real, label="Target n")
            ax.plot(lam_fit, fitted_nk.real, "--", label="Fitted n")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Real refractive index")
            ax.set_title("Fit of a synthetic oscillator spectrum")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "A good fit means the dashed curve tracks the target closely and `result.success` is `True`. The fitted dictionary can be reused directly with `multi_oscillator` or as an extrapolation model elsewhere in the library.",
            "pitfalls": [
                "Bad initial guesses can trap the optimizer in poor local minima",
                "If a fit looks unstable, add bounds or reduce the number of free oscillator terms",
                "Always inspect both `n` and `k` when you fit measured data",
            ],
            "next": [
                "Fit only one oscillator family to see where the model becomes too rigid",
                "Pass custom `weights` if matching the imaginary part is more important than the real part",
            ],
        },
        {
            "title": "Use effective-medium utilities and Kramers-Kronig reconstruction",
            "functions": ["nk.emt_multilayer_sphere", "nk.emt_brugg", "nk.eps_real_kkr"],
            "problem": "The effective-medium tools help when you want a compact surrogate for a layered particle or a composite medium. `eps_real_kkr` is useful when you know an absorption profile and need the dispersive real part implied by causality.",
            "parameters": [
                "`D`: shell diameters for `emt_multilayer_sphere`",
                "`Np`: refractive index of each shell, from inner to outer layer",
                "`fv_1`, `nk_1`, `nk_2`: volume fraction and refractive indices for the Bruggeman mixture",
                "`eps_imag`: imaginary dielectric component, given here as an array on the same wavelength grid",
            ],
            "outputs": [
                "`n_eff_layered`: effective index of a multilayer sphere",
                "`n_eff_mix`: Bruggeman effective index for a binary composite",
                "`eps_real`: reconstructed real dielectric function from the chosen imaginary spectrum",
            ],
            "code": """
            lam_emt = np.linspace(0.60, 2.20, 180)

            n_eff_layered = nk.emt_multilayer_sphere(
                [0.08, 0.10],                   # diameters of the inner core and outer shell
                [1.70 + 0.00j, 2.30 + 0.02j],   # refractive index of each layer
            )

            n_eff_mix = nk.emt_brugg(
                0.25,           # inclusion volume fraction
                2.4 + 0.10j,    # inclusion refractive index
                1.5 + 0.00j,    # host refractive index
            )

            eps_imag = np.exp(-((lam_emt - 1.20) / 0.25) ** 2)
            eps_real = nk.eps_real_kkr(
                lam_emt,
                eps_imag,  # imaginary dielectric spectrum sampled on lam_emt
                eps_inf=1.8,
            )

            print("Layered-sphere effective index:", n_eff_layered)
            print("Bruggeman effective index:", n_eff_mix)

            fig, ax = plt.subplots()
            ax.plot(lam_emt, eps_real, label="KK reconstructed eps'")
            ax.axhline(np.real(n_eff_mix**2), color="tab:orange", linestyle="--", label="Bruggeman eps'")
            ax.axhline(np.real(n_eff_layered**2), color="tab:green", linestyle=":", label="Layered EMT eps'")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Real dielectric constant")
            ax.set_title("Effective-medium surrogates and KK reconstruction")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The EMT functions return compact effective indices that you can feed into wave-optics or radiative-transfer models. The Kramers-Kronig reconstruction links absorption and dispersion, so a peak in the imaginary part creates a dispersive feature in the real part.",
            "pitfalls": [
                "EMT is an approximation; it is most useful when the inhomogeneity is subwavelength relative to the optical problem",
                "For `eps_real_kkr`, the imaginary spectrum must be sampled on the same wavelength grid you pass in",
                "If you use a callable imaginary part instead of an array, you should think carefully about the integration limits",
            ],
            "next": [
                "Change the filling fraction in `emt_brugg` and observe how fast the effective index moves toward the inclusion material",
                "Use a broader or narrower `eps_imag` peak to see the dispersion change in `eps_real_kkr`",
            ],
        },
        {
            "title": "Temperature-dependent VO2 response",
            "functions": ["nk.VO2"],
            "problem": "VO2 is a useful example of a phase-change material with a temperature-controlled optical response. This function blends the cold and hot optical constants internally and gives you a single temperature-dependent spectrum.",
            "parameters": [
                "`wavelength`: wavelength grid in micrometers",
                "`T`: material temperature in Celsius",
                "`film`: selects which tabulated VO2 film from the packaged dataset is used",
            ],
            "outputs": [
                "`n_cold` and `n_hot`: complex spectra at two temperatures",
                "a plot of the real refractive index change across the transition",
            ],
            "code": """
            lam_vo2 = np.linspace(0.80, 5.00, 220)

            n_cold = nk.VO2(
                lam_vo2,
                T=25,   # temperature in Celsius
                film=2, # packaged film choice from the library
            )

            n_hot = nk.VO2(
                lam_vo2,
                T=90,
                film=2,
            )

            fig, ax = plt.subplots()
            ax.plot(lam_vo2, n_cold.real, label="VO2 at 25 C")
            ax.plot(lam_vo2, n_hot.real, label="VO2 at 90 C")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Real refractive index")
            ax.set_title("Temperature-driven VO2 response")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The separation between the cold and hot curves is the optical signature of the phase transition. You can use these outputs anywhere else in the library that expects a wavelength-dependent refractive index.",
            "pitfalls": [
                "Temperatures are in Celsius, not Kelvin",
                "The output depends on the selected `film` dataset, so keep the film choice consistent when comparing cases",
            ],
            "next": [
                "Evaluate more temperatures between 25 C and 90 C to map the transition curve",
                "Feed the cold and hot spectra into `waveoptics.multilayer` to study a thermochromic coating",
            ],
        },
        {
            "title": "Offline material shortcut catalog",
            "functions": [
                "nk.CaCO3", "nk.BaSO4", "nk.BiVO4_mono_a", "nk.BiVO4_mono_b", "nk.BiVO4_mono_c",
                "nk.BiVO4", "nk.Cu2O", "nk.MgO", "nk.GSTa", "nk.GSTc", "nk.VO2M", "nk.VO2R",
                "nk.gold", "nk.silver", "nk.Cu", "nk.Al", "nk.HDPE", "nk.PDMS", "nk.PVDF", "nk.H2O",
            ],
            "problem": "Many day-to-day workflows do not need a file name or a full web lookup. The shortcut functions wrap the common packaged materials so you can request a refractive-index spectrum with one explicit call per material.",
            "parameters": [
                "`lam_short`: a short wavelength grid used here only to prove the functions run and return complex values",
                "each shortcut takes the same first argument: the wavelength grid in micrometers",
                "`VO2M` and `VO2R` also accept a `film` selector if you want a different packaged VO2 dataset",
            ],
            "outputs": [
                "`shortcut_preview`: one representative complex value per material shortcut",
                "a compact plot comparing a few real refractive-index curves",
            ],
            "code": """
            lam_short = np.linspace(0.90, 1.10, 5)
            lam_gst = np.linspace(2.70, 3.00, 5)

            shortcut_catalog = {
                "CaCO3": nk.CaCO3(lam_short),
                "BaSO4": nk.BaSO4(lam_short),
                "BiVO4_mono_a": nk.BiVO4_mono_a(lam_short),
                "BiVO4_mono_b": nk.BiVO4_mono_b(lam_short),
                "BiVO4_mono_c": nk.BiVO4_mono_c(lam_short),
                "BiVO4": nk.BiVO4(lam_short),
                "Cu2O": nk.Cu2O(lam_short),
                "MgO": nk.MgO(lam_short),
                "GSTa": nk.GSTa(lam_gst),
                "GSTc": nk.GSTc(lam_gst),
                "VO2M": nk.VO2M(lam_short, film=2),
                "VO2R": nk.VO2R(lam_short, film=2),
                "gold": nk.gold(lam_short),
                "silver": nk.silver(lam_short),
                "Cu": nk.Cu(lam_short),
                "Al": nk.Al(lam_short),
                "HDPE": nk.HDPE(lam_short),
                "PDMS": nk.PDMS(lam_short),
                "PVDF": nk.PVDF(lam_short),
                "H2O": nk.H2O(lam_short),
            }

            shortcut_preview = pd.Series(
                {name: complex(np.atleast_1d(values)[0]) for name, values in shortcut_catalog.items()},
                name="representative n + ik",
            )
            display(shortcut_preview)

            fig, ax = plt.subplots()
            for label in ("CaCO3", "BaSO4", "gold", "H2O"):
                ax.plot(lam_short, np.real(shortcut_catalog[label]), label=label)
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Real refractive index")
            ax.set_title("Examples of offline shortcut materials")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "Each shortcut returns the same kind of complex spectrum as `get_nkfile`, but with a simpler entry point. This is useful when you already know you want one of the packaged materials.",
            "pitfalls": [
                "Shortcut functions are convenient, but you should still check the valid wavelength range of the underlying data if you work near the edges",
                "For VO2 the shortcut pair `VO2M` and `VO2R` represent the cold and hot endpoint data, while `VO2` mixes them continuously with temperature",
            ],
            "next": [
                "Pick a few shortcuts and compare their imaginary parts to identify which ones are absorptive in your band of interest",
                "Use these shortcut outputs directly in `miescattering.scatter_efficiency` or `waveoptics.multilayer`",
            ],
        },
    ],
    "appendix": appendix_cells(
        "Online material lookup examples",
        "The next cell is intentionally commented out. These functions require internet access because they pull data from the refractiveindex.info database. Keep them out of your default offline workflow, but use them when you need a material that is not bundled locally.",
        """
        # lam = np.linspace(0.40, 1.60, 100)
        #
        # ri_table = nk.ri_info_data(
        #     "main",          # shelf in refractiveindex.info
        #     "SiO2",          # material name (book)
        #     "Franta-25C",    # page / dataset label
        # )
        #
        # n_web, ri_table_interp = nk.get_ri_info(
        #     lam,
        #     "main",
        #     "SiO2",
        #     "Franta-25C",
        # )
        #
        # n_sio2 = nk.SiO2(lam)
        # n_silica = nk.Silica(lam)
        # n_baf2 = nk.BaF2(lam)
        # n_tio2 = nk.TiO2(lam)
        # n_zno = nk.ZnO(lam)
        # n_al2o3 = nk.Al2O3(lam)
        # n_zns = nk.ZnS(lam)
        # n_si = nk.Si(lam)
        # n_mg = nk.Mg(lam)
        # n_pmma = nk.PMMA(lam)
        #
        # display(ri_table.head())
        """,
    ),
}


NOTEBOOK_SPECS["ref_spectra_test.ipynb"] = {
    "title": "ref_spectra",
    "summary": "This notebook shows how to work with the reference spectra in `empylib.ref_spectra`: solar and thermal sources, atmospheric transmission, photopic weighting, spectral averages, and plotting helpers.",
    "goals": [
        "load packaged reference spectra and compare them on a common wavelength grid",
        "compute solar and thermal averages of a spectral property",
        "plot spectral data with the library helper instead of hand-building every figure",
    ],
    "setup": """
    import empylib.ref_spectra as ref
    """,
    "sections": [
        {
            "title": "Load reference spectra and compare the main sources",
            "functions": ["ref.read_spectrafile", "ref.AM15", "ref.T_atmosphere", "ref.T_atmosphere_hemi", "ref.Bplanck"],
            "problem": "Many optical calculations need a weighting spectrum or an atmospheric transmission curve. This section shows how to fetch the most common packaged references on a shared wavelength grid.",
            "parameters": [
                "`wavelength`: wavelength grid in micrometers used for all interpolated spectra",
                "`MaterialName='T_atmosphere.txt'`: packaged text file loaded with `read_spectrafile`",
                "`spectra_type`: choose `'global'` or `'direct'` for AM1.5",
                "`beta_tilt`: collector tilt angle in degrees for hemispherical atmosphere",
                "`T`: blackbody temperature in Kelvin for `Bplanck`",
            ],
            "outputs": [
                "interpolated arrays for AM1.5, atmospheric transmission, and Planck emission",
                "`atm_table`: source table used by the atmospheric file reader",
            ],
            "code": """
            lam = np.linspace(0.30, 25.0, 400)

            atm_values, atm_table = ref.read_spectrafile(
                lam,
                "T_atmosphere.txt",  # packaged file inside empylib/ref_spectra_data
                get_from_local_path=True,
                return_data=True,
            )

            am15_global = ref.AM15(
                lam,
                spectra_type="global",  # diffuse + direct irradiance
            )

            am15_direct = ref.AM15(
                lam,
                spectra_type="direct",  # direct normal irradiance
            )

            t_atm = ref.T_atmosphere(lam)
            t_atm_hemi = ref.T_atmosphere_hemi(
                lam,
                beta_tilt=30,  # receiver tilt in degrees
            )

            b_300 = ref.Bplanck(
                lam,
                T=300,                # blackbody temperature in Kelvin
                unit="wavelength",    # return B_lambda
            )

            display(atm_table.head())

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(lam, am15_global / am15_global.max(), label="AM1.5 global")
            ax.plot(lam, am15_direct / am15_direct.max(), label="AM1.5 direct")
            ax.plot(lam, t_atm, label="Atmospheric transmission")
            ax.plot(lam, t_atm_hemi, label="Hemispherical atmosphere")
            ax.plot(lam, b_300 / np.nanmax(b_300), label="Planck spectrum at 300 K")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Normalized response")
            ax.set_title("Reference spectra used across EMPI Lib")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The solar spectra dominate at short wavelengths, while the 300 K Planck curve lives in the thermal infrared. The atmosphere functions are already dimensionless transmission factors, so they should stay between 0 and 1.",
            "pitfalls": [
                "Do not mix normalized curves with physically scaled curves when you need an energy balance",
                "The hemispherical atmosphere depends on tilt, so keep that parameter consistent with your geometry",
                "For thermal emission you usually want wavelengths extending well into the infrared",
            ],
            "next": [
                "Change `T` in `Bplanck` to 500 K or 1000 K and watch the peak shift",
                "Overlay `atm_values` and `t_atm` to confirm the file reader and helper agree",
            ],
        },
        {
            "title": "Compute photopic and spectrum-weighted averages",
            "functions": ["ref.yCIE_lum", "ref.spectral_average"],
            "problem": "Once you have a wavelength-dependent property, you often need a single weighted number: visible brightness, solar absorptance, or thermal emittance. This section shows the common workflow.",
            "parameters": [
                "`lam_visible`: visible wavelength range for the CIE photopic luminosity function",
                "`lam_property`: wavelength range of the property you want to average",
                "`absorptance`: spectral property sampled on `lam_property`",
                "`spectrum='solar'` or `'thermal'`: chooses the weighting kernel used by `spectral_average`",
            ],
            "outputs": [
                "`y_lum`: photopic sensitivity function",
                "`solar_average` and `thermal_average`: scalar weighted averages",
            ],
            "code": """
            lam_visible = np.linspace(0.38, 0.78, 200)
            y_lum = ref.yCIE_lum(lam_visible)

            lam_property = np.linspace(0.30, 25.0, 400)
            absorptance = 0.15 + 0.75 * np.exp(-((lam_property - 1.10) / 0.60) ** 2)

            solar_average = ref.spectral_average(
                lam_property,
                absorptance,
                spectrum="solar",  # AM1.5-weighted average
            )

            thermal_average = ref.spectral_average(
                lam_property,
                absorptance,
                spectrum="thermal", # Planck-weighted average
                T=500,              # emitter temperature in Kelvin
            )

            print("Solar-weighted average:", solar_average)
            print("Thermal-weighted average at 500 K:", thermal_average)

            fig, ax = plt.subplots()
            ax.plot(lam_visible, y_lum, color="tab:green", label="y_CIE")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Relative response")
            ax.set_title("Photopic luminosity function")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The luminosity curve peaks near green wavelengths because that is where the photopic human eye is most sensitive. The two scalar averages differ because solar weighting and 500 K thermal weighting emphasize very different spectral regions.",
            "pitfalls": [
                "A weighted average is only meaningful if your wavelength grid covers the range where the weighting spectrum is important",
                "If you use `spectrum='thermal'`, always pass the temperature you actually care about",
            ],
            "next": [
                "Replace `absorptance` with a measured reflectance or transmittance spectrum",
                "Compare thermal averages at 300 K, 500 K, and 1000 K",
            ],
        },
        {
            "title": "Use `plot_spectra` for quick publication-style comparisons",
            "functions": ["ref.plot_spectra"],
            "problem": "You will often want to show more than one spectral quantity on the same axes with the built-in solar and thermal backgrounds. `plot_spectra` gives you a consistent plotting style for that job.",
            "parameters": [
                "each curve is passed as `(wavelength, values, style_dict)`",
                "`ylabel`: label for the plotted spectral quantity",
                "`title`: figure title",
                "`show_background_legend=True`: include the background guides in the legend",
            ],
            "outputs": [
                "`fig, ax`: the matplotlib handles returned by `plot_spectra`",
                "a ready-to-edit figure with a consistent reference-spectrum background",
            ],
            "code": """
            fig, ax = ref.plot_spectra(
                (
                    lam_property,
                    absorptance,
                    {
                        "label": "Absorptance",
                        "color": "black",
                    },
                ),
                (
                    lam_property,
                    1 - absorptance,
                    {
                        "label": "Reflectance surrogate",
                        "color": "tab:blue",
                    },
                ),
                ylabel="Dimensionless spectrum",
                title="Using plot_spectra with standard backgrounds",
                show_background_legend=True,
            )
            plt.show()
            """,
            "read": "This helper is convenient when you want consistent context around a spectrum, especially if your readers need to see where the solar and thermal bands sit relative to your data.",
            "pitfalls": [
                "The style dictionaries are passed directly to matplotlib, so invalid keys will raise plotting errors",
                "If you compare spectra sampled on different wavelength grids, keep the units consistent",
            ],
            "next": [
                "Add a third curve such as transmittance or emissivity",
                "Customize the line styles and colors in the per-curve dictionaries",
            ],
        },
    ],
}


NOTEBOOK_SPECS["waveoptics_test.ipynb"] = {
    "title": "waveoptics",
    "summary": "This notebook teaches the thin-film optics tools in `empylib.waveoptics`, from a single interface to coherent and incoherent multilayer stacks.",
    "goals": [
        "compute Fresnel coefficients at a single interface",
        "distinguish coherent and incoherent multilayer modeling",
        "use `snell` to verify transmitted angles in layered systems",
    ],
    "setup": """
    import empylib.waveoptics as wv
    """,
    "sections": [
        {
            "title": "Single interface and transmitted angle",
            "functions": ["wv.interface", "wv.snell"],
            "problem": "Start with the simplest optical boundary: one medium above, one medium below. The library gives you reflectance, transmittance, and the complex field coefficients, while `snell` lets you verify the transmitted angle explicitly.",
            "parameters": [
                "`N_above`: refractive index of the incident medium",
                "`N_below`: refractive index of the transmitted medium; it can be scalar or wavelength-dependent",
                "`aoi`: angle of incidence in radians; here we pass three angles at once",
                "`polarization='TM'`: choose between TE and TM polarization",
            ],
            "outputs": [
                "`R` and `T`: interface reflectance and transmittance",
                "`r` and `t`: complex field reflection and transmission coefficients",
                "`theta_t`: transmitted angle from Snell's law",
            ],
            "code": """
            lam = np.linspace(0.40, 1.00, 160)
            aoi = np.radians([0.0, 30.0, 60.0])
            n_below = 1.50 + 0.01j + 0 * lam

            R, T, r, t = wv.interface(
                1.00,          # N_above: refractive index of the incident medium
                n_below,       # N_below: wavelength-dependent refractive index below the interface
                aoi=aoi,       # incidence angle(s) in radians
                polarization="TM",
            )

            theta_t = wv.snell(
                1.00,
                1.50,
                np.radians(35.0),  # incident angle in radians
            )

            print("R shape:", np.asarray(R).shape)
            print("T shape:", np.asarray(T).shape)
            print("Transmitted angle at 35 deg incidence:", np.degrees(theta_t), "deg")

            fig, ax = plt.subplots()
            ax.plot(lam, R[0], label="R at 0 deg")
            ax.plot(lam, R[1], label="R at 30 deg")
            ax.plot(lam, R[2], label="R at 60 deg")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Reflectance")
            ax.set_title("Single-interface Fresnel response")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "At larger incidence angles the TM response changes strongly, which is why angle and polarization control matter so much in thin-film design. The output shapes also show that the function can evaluate several angles in one call.",
            "pitfalls": [
                "Angles are in radians, not degrees",
                "If one refractive index is wavelength-dependent and the other is scalar, the library broadcasts the scalar automatically",
                "For absorbing media, `r` and `t` are complex amplitudes; inspect `R` and `T` for energy flow",
            ],
            "next": [
                "Switch `polarization` to `'TE'` and compare the angle dependence",
                "Use a complex `N_above` to model an absorbing incident medium",
            ],
        },
        {
            "title": "Coherent multilayer stack",
            "functions": ["wv.multilayer"],
            "problem": "When your layers are optically coherent, interference between multiple reflections shapes the spectrum. This example builds a two-layer stack and computes its coherent reflectance and transmittance.",
            "parameters": [
                "`N_layers`: list of layer refractive indices from top to bottom",
                "`thickness`: thickness of each layer in micrometers",
                "`N_above` and `N_below`: refractive index of the incident and substrate media",
                "`aoi`: angle of incidence in radians",
            ],
            "outputs": [
                "`R` and `T`: coherent reflectance and transmittance spectra",
                "`r` and `t`: complex stack amplitudes",
            ],
            "code": """
            lam_stack = np.linspace(0.40, 1.00, 180)
            n_layers = [
                1.45 + 0.00j + 0 * lam_stack,
                2.10 + 0.02j + 0 * lam_stack,
            ]
            thickness = [0.12, 0.05]

            R_coh, T_coh, r_coh, t_coh = wv.multilayer(
                lam_stack,
                aoi=np.radians(15.0),   # incidence angle in radians
                N_layers=n_layers,      # list of layer refractive indices
                thickness=thickness,    # layer thicknesses in micrometers
                N_above=1.00,           # incident medium
                N_below=1.52,           # substrate
            )

            print("First coherent R/T pair:", R_coh[0], T_coh[0])

            fig, ax = plt.subplots()
            ax.plot(lam_stack, R_coh, label="R")
            ax.plot(lam_stack, T_coh, label="T")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Flux")
            ax.set_title("Coherent multilayer film")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The oscillatory structure is the interference signature of a coherent stack. Layer thickness and optical thickness control where the extrema appear.",
            "pitfalls": [
                "The `N_layers` and `thickness` lists must have the same number of entries",
                "If you accidentally pass thickness in nanometers, the interference pattern will be completely wrong",
            ],
            "next": [
                "Double one layer thickness and watch the fringes shift",
                "Replace one layer index with a real material from `nklib`",
            ],
        },
        {
            "title": "Partially incoherent multilayer stack",
            "functions": ["wv.incoh_multilayer"],
            "problem": "Real coatings can become incoherent when the coherence length is shorter than the optical path differences in the stack. `incoh_multilayer` approximates that regime.",
            "parameters": [
                "`coh_length`: coherence length in micrometers used to decide whether interference survives",
                "the remaining arguments mirror the coherent stack API so you can compare the two models directly",
            ],
            "outputs": [
                "`R_incoh` and `T_incoh`: incoherent reflectance and transmittance spectra",
                "a side-by-side comparison against the coherent result",
            ],
            "code": """
            R_incoh, T_incoh = wv.incoh_multilayer(
                lam_stack,
                N_layers=n_layers,
                thickness=thickness,
                aoi=np.radians(15.0),
                N_above=1.00,
                N_below=1.52,
                coh_length=0.03,  # coherence length in micrometers
            )

            fig, ax = plt.subplots()
            ax.plot(lam_stack, R_coh, "--", label="R coherent")
            ax.plot(lam_stack, R_incoh, label="R incoherent")
            ax.plot(lam_stack, T_coh, "--", label="T coherent")
            ax.plot(lam_stack, T_incoh, label="T incoherent")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Flux")
            ax.set_title("Coherent versus incoherent multilayer modeling")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The incoherent spectrum is smoother because some interference detail has been averaged out. This is often closer to what you measure in rough, thick, or broadband-illuminated stacks.",
            "pitfalls": [
                "A very short `coh_length` can wash out structure you may still want to resolve",
                "A coherence model is still an approximation; if surface roughness or scattering is strong, a wave-only model may not be enough",
            ],
            "next": [
                "Increase `coh_length` to see the result move back toward the coherent case",
                "Compare the same stack at a larger angle of incidence",
            ],
        },
    ],
}


NOTEBOOK_SPECS["miescattering_test.ipynb"] = {
    "title": "miescattering",
    "summary": "This notebook teaches the single-particle and ensemble-scattering tools in `empylib.miescattering`, from efficiencies and coefficients to angular phase functions and packed-particle corrections.",
    "goals": [
        "compute efficiency factors for single and coated spheres",
        "inspect scattering coefficients and angular amplitudes explicitly",
        "build ensemble quantities from particle size distributions and phase functions",
    ],
    "setup": """
    import empylib.miescattering as mie
    """,
    "sections": [
        {
            "title": "Single homogeneous sphere",
            "functions": ["mie.scatter_efficiency"],
            "problem": "The most common entry point is the scattering efficiency of one sphere embedded in a host medium. This gives absorption, scattering, and asymmetry factor on a wavelength grid.",
            "parameters": [
                "`wavelength`: wavelength grid in micrometers",
                "`Nh`: refractive index of the host medium",
                "`Np_shells`: refractive index of the particle; a single scalar or wavelength-dependent spectrum for a homogeneous sphere",
                "`D`: particle diameter in micrometers",
            ],
            "outputs": [
                "`qabs`: absorption efficiency",
                "`qsca`: scattering efficiency",
                "`gcos`: asymmetry factor",
            ],
            "code": """
            lam = np.linspace(0.40, 1.00, 121)

            qabs, qsca, gcos = mie.scatter_efficiency(
                lam,
                1.50,                     # host refractive index
                2.35 + 0.01j + 0 * lam,  # particle refractive index
                0.30,                     # particle diameter in micrometers
            )

            print("qabs shape:", qabs.shape)
            print("g range:", float(np.min(gcos)), float(np.max(gcos)))

            fig, ax = plt.subplots()
            ax.plot(lam, qabs, label="Qabs")
            ax.plot(lam, qsca, label="Qsca")
            ax.plot(lam, gcos, label="g")
            ax.set_xlabel("Wavelength (um)")
            ax.set_title("Single-sphere scattering efficiencies")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The efficiency factors are dimensionless and tell you how strongly the particle interacts with light relative to its geometric cross section. The asymmetry factor `g` stays between `-1` and `1`.",
            "pitfalls": [
                "Diameter is in micrometers",
                "The host and particle inputs can be scalar or wavelength-dependent, but they must still be physically compatible over the same spectral range",
            ],
            "next": [
                "Change the diameter to see how resonances move with particle size",
                "Replace the constant particle index with a real material from `nklib`",
            ],
        },
        {
            "title": "Coated sphere and Mie coefficients",
            "functions": ["mie.scatter_efficiency", "mie.scatter_coefficients"],
            "problem": "A core-shell particle uses the same public API, but now the particle index and diameter are lists ordered from inner core to outer shell. You can also request the full coefficient arrays for detailed analysis.",
            "parameters": [
                "`Np_shells`: list of refractive indices, one per shell from center outward",
                "`D`: list or array of shell diameters, again from center outward",
                "`scatter_coefficients`: returns the `a_n` and `b_n` coefficient arrays instead of aggregated efficiencies",
            ],
            "outputs": [
                "`qabs_cs`, `qsca_cs`, `gcos_cs`: efficiency factors for the coated particle",
                "`an`, `bn`: Mie coefficient arrays",
            ],
            "code": """
            np_shells = [
                1.80 + 0.00j + 0 * lam,
                2.35 + 0.01j + 0 * lam,
            ]
            d_shells = np.array([0.18, 0.30])

            qabs_cs, qsca_cs, gcos_cs = mie.scatter_efficiency(
                lam,
                1.50,
                np_shells,  # core and shell refractive indices
                d_shells,   # core and outer-shell diameters
            )

            an, bn = mie.scatter_coefficients(
                lam,
                1.50,
                np_shells,
                d_shells,
            )

            print("Coefficient array shape:", an.shape)
            print("First coated-sphere g value:", gcos_cs[0])
            """,
            "read": "This is the same user-facing workflow as the homogeneous sphere, but the particle is now specified as a radial stack. The coefficient arrays are useful when you need mode-by-mode diagnostics.",
            "pitfalls": [
                "The shell diameters must increase from the core to the outer shell",
                "Keep the shell-ordering of `Np_shells` and `D` consistent",
            ],
            "next": [
                "Add a third shell to explore a multilayer sphere",
                "Plot the magnitude of selected `a_n` and `b_n` modes versus wavelength",
            ],
        },
        {
            "title": "Angular amplitudes and Stokes / Mueller terms",
            "functions": ["mie.scatter_amplitude", "mie.scatter_stokes"],
            "problem": "If you care about angular redistribution or polarization, you need more than integrated efficiencies. These functions return the scattering amplitudes and the Mueller-matrix terms on an angle grid.",
            "parameters": [
                "`theta`: scattering angle grid in radians",
                "`scatter_amplitude`: returns `S1` and `S2`",
                "`scatter_stokes`: returns `S11`, `S12`, `S33`, and `S34`",
            ],
            "outputs": [
                "`s1`, `s2`: amplitude functions",
                "`s11`, `s12`, `s33`, `s34`: angular Stokes / Mueller terms",
            ],
            "code": """
            theta = np.linspace(0.0, np.pi, 181)

            s1, s2 = mie.scatter_amplitude(
                lam,
                1.50,
                2.35 + 0.01j + 0 * lam,
                0.30,
                theta=theta,  # scattering angles in radians
            )

            s11, s12, s33, s34 = mie.scatter_stokes(
                lam,
                1.50,
                2.35 + 0.01j + 0 * lam,
                0.30,
                theta=theta,
            )

            print("S1 shape:", s1.shape)
            print("S11 shape:", s11.shape)

            fig, ax = plt.subplots()
            wavelength_index = 20
            ax.plot(np.degrees(theta), s11[:, wavelength_index], label="S11")
            ax.plot(np.degrees(theta), s12[:, wavelength_index], label="S12")
            ax.set_xlabel("Scattering angle (deg)")
            ax.set_title("Angular scattering response at one wavelength sample")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The angular functions resolve how the particle redistributes light by direction and polarization. This is the right level of detail when you need a phase function for radiative transfer or a polarization observable.",
            "pitfalls": [
                "Angles are in radians on input even if you plot them in degrees",
                "The returned arrays are angle by wavelength, so use the correct axis when slicing",
            ],
            "next": [
                "Compare two wavelengths at the same particle size",
                "Use the angular data to build a custom phase-function diagnostic",
            ],
        },
        {
            "title": "Phase functions and ensemble-averaged scattering",
            "functions": ["mie.phase_scatt_HG", "mie.scatter_from_phase_function", "mie.structure_factor_PY", "mie.phase_scatt_ensemble", "mie.cross_section_ensemble"],
            "problem": "Real particulate media usually contain a size distribution, and dense systems can also show dependent scattering. This section shows how to move from a simple phase function to a distributed ensemble and then to ensemble cross sections.",
            "parameters": [
                "`d_bins`: particle diameters used to describe the distribution",
                "`size_dist`: relative weights for each diameter bin",
                "`fv`: particle volume fraction used by the dependent-scattering corrections",
                "`phase_function=True`: request the ensemble phase function DataFrame from `cross_section_ensemble`",
            ],
            "outputs": [
                "`hg_phase`: Henyey-Greenstein phase function",
                "`qsca_hg`, `g_hg`: scattering quantities recovered from that phase function",
                "`structure_factor`: Percus-Yevick structure factor",
                "`phase_ensemble`: ensemble phase function",
                "`cabs`, `csca`, `g_av`, `phase_df`: ensemble-averaged optical quantities",
            ],
            "code": """
            d_bins = np.linspace(0.20, 0.40, 7)
            size_dist = np.exp(-((d_bins - 0.30) / 0.05) ** 2)
            size_dist = size_dist / size_dist.sum()

            hg_phase = mie.phase_scatt_HG(
                lam,
                np.full(lam.size, 0.7),  # asymmetry factor as a wavelength-dependent array
                qsca=np.ones_like(lam),  # normalization helper for the HG phase function
                theta=theta,
            )

            qsca_hg, g_hg = mie.scatter_from_phase_function(hg_phase)

            structure_factor = mie.structure_factor_PY(
                lam,
                1.50,
                d_bins,
                fv=0.08,              # particle volume fraction
                theta=theta,
                size_dist=size_dist,  # distribution weights over d_bins
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
                phase_function=True,  # also return the phase function DataFrame
            )

            print("HG-derived qsca and g:", qsca_hg[:3], g_hg[:3])
            display(phase_df.head())

            fig, ax = plt.subplots()
            ax.plot(lam, cabs, label="Cabs")
            ax.plot(lam, csca, label="Csca")
            ax.plot(lam, g_av, label="g")
            ax.set_xlabel("Wavelength (um)")
            ax.set_title("Ensemble-averaged optical response")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The ensemble functions package several common particulate-medium tasks: approximate phase functions, dependent-scattering corrections, and size-distribution averaging. The returned DataFrame is convenient when you need a tabular phase function downstream.",
            "pitfalls": [
                "The size distribution weights should be normalized if you want them to behave as probabilities",
                "Dense suspensions can be sensitive to the volume fraction and dependent-scattering model you choose",
                "If you request `phase_function=True`, remember that the phase-function output is a DataFrame in addition to the scalar spectra",
            ],
            "next": [
                "Replace the Gaussian size distribution with a log-normal one",
                "Compare the ensemble result with a single-diameter result to see how much the size spread matters",
            ],
        },
    ],
}


NOTEBOOK_SPECS["rad_transfer_test.ipynb"] = {
    "title": "rad_transfer",
    "summary": "This notebook teaches the slab-radiative-transfer tools in `empylib.rad_transfer`, from a fast Beer-Lambert estimate to the adding-doubling method with either sphere-derived or user-supplied scattering inputs.",
    "goals": [
        "compute slab transmission with the Beer-Lambert approximation",
        "switch to `adm_sphere` when particle scattering matters",
        "use `adm` directly when you already know absorption, scattering, and phase behavior",
    ],
    "setup": """
    import empylib.miescattering as mie
    import empylib.rad_transfer as rt
    """,
    "sections": [
        {
            "title": "Beer-Lambert slab with embedded particles",
            "functions": ["rt.T_beer_lambert"],
            "problem": "Start with the quickest estimate: absorption and single-pass losses in a particle-filled slab, including interface effects through the slab boundaries.",
            "parameters": [
                "`Nh`: refractive index of the host medium",
                "`Np`: particle refractive index",
                "`D`: particle diameter in micrometers",
                "`fv`: particle volume fraction",
                "`tfilm`: slab thickness in micrometers",
                "`theta`: incidence angle in degrees for this API",
                "`Nup` and `Ndw`: refractive index above and below the slab",
            ],
            "outputs": [
                "a DataFrame with total and separated transmission / reflection channels",
            ],
            "code": """
            lam = np.linspace(0.40, 1.00, 121)
            nh = 1.49 + 0.00j + 0 * lam
            npart = 2.35 + 0.02j + 0 * lam

            beer_result = rt.T_beer_lambert(
                lam,
                nh,
                npart,
                0.30,    # particle diameter in micrometers
                fv=0.06, # particle volume fraction
                tfilm=0.50,
                theta=8.0, # angle in degrees
                Nup=1.00,
                Ndw=1.52,
            )

            display(beer_result.head())

            fig, ax = plt.subplots()
            for col in beer_result.columns:
                ax.plot(beer_result.index, beer_result[col], label=col)
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Flux")
            ax.set_title("Beer-Lambert slab with embedded particles")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "The DataFrame columns separate total, specular, and diffuse flux channels depending on what the approximation can predict. This is the fastest workflow when you mainly want a first estimate of slab transmittance.",
            "pitfalls": [
                "This API expects `theta` in degrees, not radians",
                "Beer-Lambert is an approximation; once multiple scattering matters, it can become too optimistic",
            ],
            "next": [
                "Increase `fv` or `tfilm` and compare how strongly transmission collapses",
                "Switch to `adm_sphere` for the same slab to see the effect of multiple scattering",
            ],
        },
        {
            "title": "Adding-doubling for spherical particles",
            "functions": ["rt.adm_sphere"],
            "problem": "When the slab is scattering strongly, the adding-doubling method is more appropriate. `adm_sphere` keeps the convenient particle specification and computes the internal transport quantities for you.",
            "parameters": [
                "`dependent_scatt=False`: disable structure-factor corrections in this example",
                "`effective_medium=True`: include the host/particle effective-medium correction for the slab index",
                "`use_phase_fun=False`: use the asymmetry factor instead of the full phase function in this example",
            ],
            "outputs": [
                "an adding-doubling result DataFrame with total, specular, and diffuse components",
            ],
            "code": """
            adm_sphere_result = rt.adm_sphere(
                lam,
                nh,
                npart,
                0.30,
                fv=0.06,
                tfilm=0.50,
                Nup=1.00,
                Ndw=1.52,
                dependent_scatt=False,
                effective_medium=True,
                use_phase_fun=False,
            )

            display(adm_sphere_result.head())

            fig, ax = plt.subplots()
            for col in adm_sphere_result.columns:
                ax.plot(adm_sphere_result.index, adm_sphere_result[col], label=col)
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Flux")
            ax.set_title("Adding-doubling model with spherical particles")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "Compared with Beer-Lambert, adding-doubling can redistribute energy between specular and diffuse channels through multiple scattering. This is usually the safer model when particle scattering is not negligible.",
            "pitfalls": [
                "Adding-doubling is more informative, but it also asks you to think harder about the phase-function approximation",
                "Keep the host, particle, and film inputs consistent with the physical slab you want to model",
            ],
            "next": [
                "Enable `use_phase_fun=True` and compare the result with the asymmetry-factor approximation",
                "Turn on `dependent_scatt=True` for a denser medium",
            ],
        },
        {
            "title": "Direct adding-doubling from user-supplied scattering data",
            "functions": ["rt.adm"],
            "problem": "Sometimes you already know the absorption coefficient, scattering coefficient, and either an asymmetry factor or a full phase function. In that case you can bypass the sphere model and call `adm` directly.",
            "parameters": [
                "`k_sca` and `k_abs`: scattering and absorption coefficients on the wavelength grid",
                "`gcos`: asymmetry factor, used in the first direct ADM call",
                "`phase_fun`: full phase function, used in the second direct ADM call",
                "`Nh`: refractive index of the slab host medium",
            ],
            "outputs": [
                "`adm_from_g` and `adm_from_phase`: two DataFrames that differ only in how the angular information is supplied",
            ],
            "code": """
            k_sca = 0.8 * np.exp(-((lam - 0.65) / 0.16) ** 2)
            k_abs = 0.2 * np.exp(-((lam - 0.82) / 0.12) ** 2)
            gcos = np.clip(0.2 + 0.6 * (lam - lam.min()) / (lam.max() - lam.min()), 0, 0.95)

            adm_from_g = rt.adm(
                lam,
                tfilm=0.30,
                k_sca=k_sca,
                k_abs=k_abs,
                Nh=nh,
                gcos=gcos,  # asymmetry factor description of angular scattering
                Nup=1.00,
                Ndw=1.52,
            )

            phase_fun = mie.phase_scatt_HG(
                lam,
                gcos=np.full(lam.size, 0.75),
                qsca=np.ones_like(lam),
                theta=np.linspace(0.0, np.pi, 181),
            )

            adm_from_phase = rt.adm(
                lam,
                tfilm=0.30,
                k_sca=k_sca,
                k_abs=k_abs,
                Nh=nh,
                phase_fun=phase_fun,  # full angle-resolved phase function
                Nup=1.00,
                Ndw=1.52,
            )

            fig, ax = plt.subplots()
            ax.plot(lam, adm_from_g["Ttot"], label="Ttot from g")
            ax.plot(lam, adm_from_phase["Ttot"], label="Ttot from phase function")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Total transmission")
            ax.set_title("ADM with g-only versus full phase function")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "This direct API is useful when your scattering information comes from measurement, another solver, or a custom phase-function model. The comparison shows how much difference there is between a compact `g` description and a fully resolved angular phase function.",
            "pitfalls": [
                "Pass either `gcos` or `phase_fun` according to the model you want; do not confuse a scalar asymmetry factor with a full phase function",
                "The transport coefficients must be sampled on the same wavelength grid as the refractive-index inputs",
            ],
            "next": [
                "Replace the HG phase function with one built from `miescattering.phase_scatt_ensemble`",
                "Inspect `Rdif` and `Tdif` in addition to `Ttot` to understand where the changes come from",
            ],
        },
    ],
}


NOTEBOOK_SPECS["spectrometry_test.ipynb"] = {
    "title": "spectrometry",
    "summary": "This notebook teaches the UV-Vis reader and sample-aggregation tools in `empylib.spectrometry`. The examples use the packaged sample files in `docs/` so you can reproduce the workflow offline.",
    "goals": [
        "read a single UV-Vis export file into a tidy DataFrame",
        "scan a folder to discover available sample names automatically",
        "aggregate all measurements belonging to one sample into a combined reflectance / transmittance table",
    ],
    "setup": """
    import empylib.spectrometry as spec

    DOCS = ROOT / "docs"
    UVVIS_DIR = DOCS / "uvvis_sample_data"
    PERKIN_PATH = DOCS / "PerkinElmer_sample.asc"
    """,
    "sections": [
        {
            "title": "Read one UV-Vis file",
            "functions": ["spec.read_uvvis"],
            "problem": "The basic workflow starts with a single exported file. `read_uvvis` detects the vendor format, parses the data, converts wavelengths to micrometers, and scales percent columns to fractions when needed.",
            "parameters": [
                "`path`: path to the UV-Vis export file",
                "`vendor`: optional manual override if the extension is ambiguous",
                "`col1_name` and `col2_name`: optional manual column names if the file headers are incomplete",
            ],
            "outputs": [
                "a DataFrame indexed by wavelength in micrometers",
                "metadata in `df.attrs`, such as instrument and sample name",
            ],
            "code": """
            uvvis_single = spec.read_uvvis(
                str(PERKIN_PATH),  # packaged PerkinElmer example file
            )

            display(uvvis_single.head())
            print("Instrument:", uvvis_single.attrs.get("instrument"))
            print("Sample name:", uvvis_single.attrs.get("sample_name"))
            """,
            "read": "The index is already in micrometers and the measurement column is scaled to a 0-1 fraction when the raw file looked like percent data. This is the right format for the rest of the library.",
            "pitfalls": [
                "If the instrument headers are incomplete, provide `col1_name` or `col2_name` manually",
                "The function infers the vendor from the file extension when `vendor` is omitted",
            ],
            "next": [
                "Read one of the sample files inside `UVVIS_DIR` and compare the metadata",
                "Override `col2_name` to make the output column label match your preferred naming convention",
            ],
        },
        {
            "title": "Find the samples available in a folder",
            "functions": ["spec.find_uvvis_samples"],
            "problem": "Large UV-Vis campaigns often generate several files per sample. `find_uvvis_samples` scans the naming patterns and returns the unique sample names it can infer.",
            "parameters": [
                "`search_dirs`: folders scanned for UV-Vis files",
                "`tags`: optional measurement tags such as `Rtot`, `Ttot`, `Rspec`, and `Tspec`",
                "`aliases`: optional alternate prefixes if your instrument naming differs from the defaults",
            ],
            "outputs": [
                "`sample_names`: list of discovered sample identifiers",
            ],
            "code": """
            sample_names = spec.find_uvvis_samples(
                search_dirs=[str(UVVIS_DIR)],
            )

            print("Discovered samples:")
            for name in sample_names:
                print("-", name)
            """,
            "read": "The returned names are the sample identifiers inferred from the filenames after removing the measurement tag prefix. This is useful when you want a notebook or script to discover data automatically instead of hardcoding sample names.",
            "pitfalls": [
                "This helper depends on filename patterns, so inconsistent naming conventions can hide valid files",
                "If your tags differ from the defaults, pass custom `tags` and `aliases`",
            ],
            "next": [
                "Restrict the scan to a smaller tag list if you only care about reflectance or only about transmittance",
                "Use the discovered sample names in a loop with `sample_uvvis`",
            ],
        },
        {
            "title": "Aggregate all files for one sample",
            "functions": ["spec.sample_uvvis"],
            "problem": "A full sample usually includes several measurements: total, specular, and diffuse reflectance or transmittance. `sample_uvvis` reads all matching files, interpolates them onto a common grid, and combines them into one DataFrame.",
            "parameters": [
                "`sample`: sample name as it appears in the filenames after the measurement tag",
                "`search_dirs`: folders where the measurement files live",
                "`ref_standard='spectralon'`: reflectance standard used for the correction of reflectance channels",
            ],
            "outputs": [
                "a DataFrame with columns such as `Rtot`, `Rspec`, `Rdif`, `Ttot`, `Tspec`, and `Tdif` when available",
                "sample metadata stored in `dataset.attrs`",
            ],
            "code": """
            uvvis_sample = spec.sample_uvvis(
                sample="PMMApaint_CaCO3_15vv",
                search_dirs=[str(UVVIS_DIR)],
            )

            display(uvvis_sample.head())
            print("Columns:", list(uvvis_sample.columns))
            print("Sample attribute:", uvvis_sample.attrs.get("sample_name"))

            fig, ax = plt.subplots()
            for col in uvvis_sample.columns:
                ax.plot(uvvis_sample.index, uvvis_sample[col], label=col)
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Measured fraction")
            ax.set_title("Aggregated UV-Vis sample")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
            """,
            "read": "This DataFrame is the practical endpoint of a UV-Vis import workflow: one wavelength index, several corrected measurement channels, and consistent naming that is easy to feed into later optical calculations.",
            "pitfalls": [
                "If one expected file is missing, the returned DataFrame may still be valid but have fewer columns",
                "The function interpolates onto the union grid of the discovered files, so small NaN patterns can appear if a measurement does not cover the full range",
            ],
            "next": [
                "Loop over all discovered sample names and store the resulting DataFrames in a dictionary",
                "Feed one reflectance spectrum into `color_system.spectrum_to_hex` to estimate the visible color",
            ],
        },
    ],
}


NOTEBOOK_SPECS["scuffem_test.ipynb"] = {
    "title": "scuffem",
    "summary": "This notebook teaches the helper functions in `empylib.scuffem` for writing spectral material files and reading SCUFF-EM output tables.",
    "goals": [
        "generate spectral input files for SCUFF-EM in a temporary working directory",
        "read packaged SCUFF-EM scattering outputs into Python objects",
        "clean raw tables before downstream analysis",
    ],
    "setup": """
    import tempfile

    import empylib.scuffem as scf

    DATA_DIR = ROOT / "docs" / "tio2_D700nm"
    """,
    "sections": [
        {
            "title": "Write spectral input files",
            "functions": ["scf.make_spectral_files"],
            "problem": "Before a SCUFF-EM run, you need an omega list and material files in the expected format. `make_spectral_files` creates those files from a wavelength grid and a material dictionary.",
            "parameters": [
                "`wavelength`: wavelength grid in micrometers",
                "`Material`: dictionary mapping material names to complex refractive-index arrays",
                "the current working directory determines where the files are written",
            ],
            "outputs": [
                "`OmegaList.dat` and one material `.dat` file per dictionary entry",
            ],
            "code": """
            lam = np.linspace(0.45, 1.10, 51)
            material_dict = {
                "demo_material": 2.20 + 0.02j + 0 * lam,
            }

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                current = Path.cwd()
                try:
                    os.chdir(tmp_path)
                    scf.make_spectral_files(
                        lam,
                        material_dict,  # name -> complex refractive-index spectrum
                    )
                finally:
                    os.chdir(current)

                omega_lines = (tmp_path / "OmegaList.dat").read_text(encoding="utf-8").splitlines()
                material_lines = (tmp_path / "demo_material.dat").read_text(encoding="utf-8").splitlines()

            print("First omega lines:", omega_lines[:3])
            print("First material lines:", material_lines[:5])
            """,
            "read": "The helper converts your wavelength-based optical data into the text files expected by SCUFF-EM. Using a temporary directory is a good habit for tutorials and tests because it leaves the repository clean.",
            "pitfalls": [
                "The files are written to the current working directory, so be explicit about where you are",
                "The material arrays must match the length of the wavelength grid",
            ],
            "next": [
                "Add a second material to the dictionary and inspect the new `.dat` file",
                "Use a real material spectrum from `nklib` instead of a constant complex value",
            ],
        },
        {
            "title": "Read SCUFF-EM scattering summaries",
            "functions": ["scf.read_scatter_PFT", "scf.read_avescatter"],
            "problem": "After a SCUFF-EM run, you usually want a structured Python object instead of raw text. These readers parse the packaged sample output and return the tables in a convenient form.",
            "parameters": [
                "`FileName`: path to a SCUFF-EM output file",
                "`read_scatter_PFT`: parse the power / force / torque style scattering summary",
                "`read_avescatter`: parse the angle-averaged scattering summary",
            ],
            "outputs": [
                "`pft_data`: parsed PFT-style content",
                "`ave_data`: parsed average-scattering content",
            ],
            "code": """
            sample_file = DATA_DIR / "tio2_D700.AVSCAT.EMTPFT"

            pft_data = scf.read_scatter_PFT(str(sample_file))
            ave_data = scf.read_avescatter(str(sample_file))

            print("PFT object type:", type(pft_data))
            print("AVE object type:", type(ave_data))
            print("First parsed PFT entry:", pft_data[0] if pft_data else "empty")
            print("First parsed AVE entry:", ave_data[0] if ave_data else "empty")
            """,
            "read": "These readers let you move from SCUFF-EM text output into Python data structures quickly, which makes later plotting or comparison work much easier.",
            "pitfalls": [
                "Use the reader that matches the file structure you have",
                "If you get an empty output, inspect the raw file to confirm the format is one the helper supports",
            ],
            "next": [
                "Convert the parsed outputs into a DataFrame for custom plotting",
                "Compare several SCUFF-EM runs by reading them into a list and stacking the outputs",
            ],
        },
        {
            "title": "Clean a parsed SCUFF-EM table",
            "functions": ["scf.clean_data"],
            "problem": "SCUFF-EM output can contain entries you want to standardize or drop before plotting. `clean_data` provides a light post-processing step.",
            "parameters": [
                "`data`: parsed SCUFF-EM output such as the result of `read_avescatter`",
                "`inplace=False`: return a cleaned copy instead of modifying the original object",
            ],
            "outputs": [
                "`cleaned_data`: sanitized version of the parsed SCUFF-EM output",
            ],
            "code": """
            cleaned_data = scf.clean_data(
                ave_data,
                inplace=False,
            )

            print("Original first entry:", ave_data[0] if ave_data else "empty")
            print("Cleaned first entry:", cleaned_data[0] if cleaned_data else "empty")
            """,
            "read": "A cleaning step is useful when you want downstream plotting or comparisons to operate on consistent column names and value types instead of raw solver text.",
            "pitfalls": [
                "If you set `inplace=True`, you are modifying the original parsed object",
                "Cleaning does not replace understanding the original file format; inspect the raw data when something looks wrong",
            ],
            "next": [
                "Wrap the cleaned output in a DataFrame for plotting",
                "Save the cleaned table to disk if you want a stable post-processed dataset",
            ],
        },
    ],
}


NOTEBOOK_SPECS["color_system_test.ipynb"] = {
    "title": "color_system",
    "summary": "This notebook teaches `empylib.color_system.spectrum_to_hex` in its three main modes: a material under a standard illuminant, a self-luminous emitter, and a material seen under a custom illuminant spectrum.",
    "goals": [
        "convert a reflectance-like spectrum into sRGB color under D65",
        "convert an emitter spectrum directly into sRGB",
        "use a custom illuminant when the viewing conditions matter",
    ],
    "setup": """
    import empylib.color_system as cs
    import empylib.ref_spectra as ref
    """,
    "sections": [
        {
            "title": "Material color under a standard illuminant",
            "functions": ["cs.spectrum_to_hex"],
            "problem": "A reflectance or transmittance spectrum is not a color by itself. To obtain a display color you must specify the viewing illuminant and convert the spectral response into XYZ and then sRGB.",
            "parameters": [
                "`wls_um`: visible wavelength grid in micrometers",
                "`values`: reflectance-like factor in the 0-1 range",
                "`source='material'`: tells the function to interpret the spectrum as a material factor rather than an emitter SPD",
                "`illuminant_name='D65'`: standard daylight illuminant used for the color appearance calculation",
            ],
            "outputs": [
                "`hex_color`: HTML-style color string",
                "`rgb01`: gamma-encoded sRGB values in the 0-1 range",
                "`rgb255`: 8-bit integer RGB triplet",
            ],
            "code": """
            lam = np.linspace(0.38, 0.78, 201)
            reflectance = 0.15 + 0.75 * np.exp(-((lam - 0.62) / 0.08) ** 2)

            hex_color, rgb01, rgb255 = cs.spectrum_to_hex(
                lam,
                reflectance,         # material reflectance-like spectrum
                source="material",   # interpret the spectrum as a material factor
                illuminant_name="D65",
            )

            print("hex_color:", hex_color)
            print("rgb01:", rgb01)
            print("rgb255:", rgb255)

            swatch = np.ones((30, 120, 3))
            swatch[..., 0] = rgb01[0]
            swatch[..., 1] = rgb01[1]
            swatch[..., 2] = rgb01[2]

            fig, ax = plt.subplots()
            ax.imshow(swatch)
            ax.set_axis_off()
            ax.set_title(f"Material color under D65: {hex_color}")
            plt.show()
            """,
            "read": "This is the standard material-color workflow for a display-oriented result. The spectrum is weighted by the illuminant and the human observer model before it is mapped to sRGB.",
            "pitfalls": [
                "The visible wavelength grid must overlap the observer color-matching functions",
                "For `source='material'`, values outside 0-1 are clipped because they are treated like reflectance or transmittance factors",
            ],
            "next": [
                "Replace the synthetic reflectance with a measured visible reflectance curve",
                "Try `illuminant_name='A'` to mimic a warmer tungsten-like viewing condition",
            ],
        },
        {
            "title": "Emitter color",
            "functions": ["cs.spectrum_to_hex"],
            "problem": "For a self-luminous source, the spectrum is interpreted as emitted power rather than a reflectance factor. No external illuminant is needed in that case.",
            "parameters": [
                "`source='emitter'`: switch to the emitter branch of the color model",
                "`emitter_units='per_um'`: the spectral power density is given per micrometer in this example",
                "`values`: emitter spectral power density, here built from a Planck spectrum",
            ],
            "outputs": [
                "the same three color outputs as before, now derived from the emitter SPD",
            ],
            "code": """
            emitter_spd = ref.Bplanck(
                lam,
                T=2600,               # blackbody-like emitter temperature in Kelvin
                unit="wavelength",
            )

            hex_emitter, rgb01_emitter, rgb255_emitter = cs.spectrum_to_hex(
                lam,
                emitter_spd,
                source="emitter",
                emitter_units="per_um",  # SPD is sampled per micrometer
            )

            print("hex_emitter:", hex_emitter)
            print("rgb01_emitter:", rgb01_emitter)
            print("rgb255_emitter:", rgb255_emitter)
            """,
            "read": "Emitter mode is the right choice for blackbodies, LEDs, or any self-luminous source. The absolute power scale does not affect the reported color because the spectrum is normalized during the color conversion.",
            "pitfalls": [
                "Do not use `source='material'` for an emitter spectrum; that would incorrectly apply an external illuminant model",
                "Set `emitter_units` correctly so the numerical integration uses the right density convention",
            ],
            "next": [
                "Increase the temperature to see the color move toward a cooler white",
                "Pass a measured lamp spectrum instead of a Planck curve",
            ],
        },
        {
            "title": "Material color under a custom illuminant",
            "functions": ["cs.spectrum_to_hex"],
            "problem": "Sometimes the standard illuminants are not what matters. This example uses AM1.5G as a custom viewing illuminant so the perceived material color reflects a solar-like source spectrum.",
            "parameters": [
                "`illuminant=(wavelength, solar_spd)`: custom illuminant given as wavelength grid plus spectral power density",
                "`illuminant_units='per_um'`: units of that custom illuminant spectrum",
                "`source='material'`: still a material factor, so the illuminant matters",
            ],
            "outputs": [
                "material color under the custom source and a plot of the illuminant spectrum used",
            ],
            "code": """
            reflectance_custom = 0.20 + 0.70 * np.exp(-((lam - 0.55) / 0.10) ** 2)
            solar = ref.AM15(
                lam,
                spectra_type="global",
            )

            hex_custom, rgb01_custom, rgb255_custom = cs.spectrum_to_hex(
                lam,
                reflectance_custom,
                source="material",
                illuminant=(lam, solar),   # custom illuminant spectrum on the same wavelength grid
                illuminant_units="per_um", # AM1.5 was sampled per micrometer
            )

            print("hex_custom:", hex_custom)
            print("rgb01_custom:", rgb01_custom)
            print("rgb255_custom:", rgb255_custom)

            fig, ax = plt.subplots()
            ax.plot(lam, solar, color="tab:orange")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Relative illuminant power")
            ax.set_title("Custom illuminant used for the material color calculation")
            ax.grid(True, alpha=0.3)
            plt.show()
            """,
            "read": "Custom illuminants matter whenever the viewing source is part of the physical question. Outdoor solar appearance, indoor lighting, or spectrally unusual lamps can all change the perceived color of the same material spectrum.",
            "pitfalls": [
                "The custom illuminant grid and the material spectrum must overlap in wavelength",
                "Keep the `illuminant_units` consistent with how the illuminant spectrum was generated",
            ],
            "next": [
                "Swap AM1.5G for a blackbody lamp spectrum and compare the resulting hex code",
                "Use a measured transmittance instead of reflectance to estimate filter color",
            ],
        },
    ],
}


def build_notebook(spec: dict) -> dict:
    cells = [
        intro_cell(spec["title"], spec["summary"], spec["goals"]),
        setup_cell(spec["setup"]),
    ]
    for section in spec["sections"]:
        cells.extend(section_cells(section))
    cells.extend(spec.get("appendix", []))
    return notebook(cells)


def main() -> None:
    for filename, spec in NOTEBOOK_SPECS.items():
        path = ROOT / filename
        path.write_text(
            json.dumps(build_notebook(spec), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {path.name}")


if __name__ == "__main__":
    main()
