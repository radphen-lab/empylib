import os
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import empylib.miescattering as mie
import empylib.rad_transfer as rt
import empylib.waveoptics as wv


LAM = 0.5
DP = 0.3

CASES = {
    "case1": {"N_back": 10.0, "nh": 1.0, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case2": {"N_back": 10.0, "nh": 1.0, "tfilm_test": 5.0, "Dp": DP, "fv": 0.5},
    "case3": {"N_back": 10.0, "nh": 1.0, "tfilm_test": 0.0, "Dp": DP, "fv": 0.5},
    "case4": {"N_back": 10.0 + 20j, "nh": 1.0, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case5": {"N_back": 10.0 + 20j, "nh": 1.0, "tfilm_test": 5.0, "Dp": DP, "fv": 0.5},
    "case6": {"N_back": 10.0 + 20j, "nh": 1.0, "tfilm_test": 0.0, "Dp": DP, "fv": 0.5},
    "case7": {"N_back": 10.0 + 50j, "nh": 1.5 + 1e-5j, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case8": {"N_back": 10.0 + 50j, "nh": 1.5 + 1e-5j, "tfilm_test": 5.0, "Dp": DP, "fv": 0.5},
    "case9": {"N_back": 10.0 + 50j, "nh": 1.5 + 1e-5j, "tfilm_test": 0.0, "Dp": DP, "fv": 0.5},
    "case10": {"N_back": 1.0, "nh": 1.5 + 1e-5j, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case11": {"N_back": 1.0, "nh": 1.5 + 1e-5j, "tfilm_test": 5.0, "Dp": DP, "fv": 0.5},
    "case12": {"N_back": 1.0, "nh": 1.5 + 1e-5j, "tfilm_test": 0.0, "Dp": DP, "fv": 0.5},
    "case13": {"N_above": 1.0, "N_back": 1.0, "nh": 1.0, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case14": {"N_above": 1.0, "N_back": 1.5, "nh": 1.5, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case15": {"N_above": 1.0, "N_back": 1.5, "nh": 1.5, "tfilm_test": 0.0, "Dp": DP, "fv": 0.5},
    "case16": {"N_above": 1.0, "N_back": 1.0, "nh": 1.5, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case17": {"N_above": 1.0, "N_back": 1.0, "nh": 1.5, "tfilm_test": 0.0, "Dp": DP, "fv": 0.5},
    "case18": {"N_above": 1.5 + 1e-5j, "N_back": 1.5 + 1e-5j, "nh": 1.5 + 1e-5j, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case19": {"N_above": 1.0, "N_back": 1.5 + 1e-5j, "nh": 1.5 + 1e-5j, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case20": {"N_above": 1.0, "N_back": 1.5 + 1e-5j, "nh": 1.5 + 1e-5j, "tfilm_test": 0.0, "Dp": DP, "fv": 0.5},
    "case21": {"N_above": 1.0, "N_back": 1.0, "nh": 1.5 + 1e-5j, "tfilm_test": 5.0, "Dp": DP, "fv": 0.0},
    "case22": {"N_above": 1.0, "N_back": 1.0, "nh": 1.5 + 1e-5j, "tfilm_test": 0.0, "Dp": DP, "fv": 0.5},
}


def _outer_above(case):
    return case.get("N_above", case["nh"])


def _reference_rt(case):
    wavelength = np.array([LAM], dtype=float)
    n_above = _outer_above(case)
    n_below = case["N_back"]
    thickness = float(case["tfilm_test"])

    if np.isclose(thickness, 0.0):
        r_ref, t_ref = wv.interface(n_above, n_below)[:2]
    else:
        r_ref, t_ref = wv.incoh_multilayer(
            wavelength=wavelength,
            N_layers=case["nh"],
            thickness=thickness * 1e3,
            N_above=n_above,
            N_below=n_below,
        )
    return float(r_ref[0]), float(t_ref[0])


class AdmSanityTest(unittest.TestCase):
    def test_01_no_scattering_limits(self):
        wavelength = np.array([LAM], dtype=float)
        atol = 2e-4

        for name, case in CASES.items():
            with self.subTest(case=name):
                n_host = np.array([case["nh"]], dtype=complex)
                n_above = np.array([_outer_above(case)], dtype=complex)
                n_below = np.array([case["N_back"]], dtype=complex)
                r_ref, t_ref = _reference_rt(case)

                out = rt.adm(
                    wavelength=wavelength,
                    thickness=float(case["tfilm_test"]),
                    k_sca=np.array([0.0], dtype=float),
                    k_abs=np.array([0.0], dtype=float),
                    N_host=n_host,
                    gcos=np.array([0.0], dtype=float),
                    N_above=n_above,
                    N_below=n_below,
                    quad_pts=16,
                )

                np.testing.assert_allclose(out["Rtot"].to_numpy(), [r_ref], atol=atol)
                np.testing.assert_allclose(out["Ttot"].to_numpy(), [t_ref], atol=atol)
                np.testing.assert_allclose(out["Rspec"].to_numpy(), [r_ref], atol=atol)
                np.testing.assert_allclose(out["Tspec"].to_numpy(), [t_ref], atol=atol)
                np.testing.assert_allclose(out["Rdif"].to_numpy(), [0.0], atol=1e-12)
                np.testing.assert_allclose(out["Tdif"].to_numpy(), [0.0], atol=1e-12)


class AdmSphereSanityTest(unittest.TestCase):
    def test_01_matched_index_limits(self):
        wavelength = np.array([LAM], dtype=float)

        for name, case in CASES.items():
            with self.subTest(case=name):
                n_host = np.array([case["nh"]], dtype=complex)
                n_above = np.array([_outer_above(case)], dtype=complex)
                n_below = np.array([case["N_back"]], dtype=complex)
                r_ref, t_ref = _reference_rt(case)

                out = rt.adm_sphere(
                    wavelength=wavelength,
                    N_host=n_host,
                    N_particle=n_host,
                    D=case["Dp"],
                    fv=float(case["fv"]),
                    thickness=float(case["tfilm_test"]),
                    N_above=n_above,
                    N_below=n_below,
                    effective_medium=True,
                    dependent_scatt=True,
                    use_phase_fun=True,
                )

                np.testing.assert_allclose(out["Rtot"].to_numpy(), [r_ref], atol=5e-6)
                np.testing.assert_allclose(out["Ttot"].to_numpy(), [t_ref], atol=5e-6)
                np.testing.assert_allclose(out["Rspec"].to_numpy(), [r_ref], atol=5e-6)
                np.testing.assert_allclose(out["Tspec"].to_numpy(), [t_ref], atol=5e-6)
                np.testing.assert_allclose(out["Rdif"].to_numpy(), [0.0], atol=1e-12)
                np.testing.assert_allclose(out["Tdif"].to_numpy(), [0.0], atol=1e-12)


class MieZeroContrastTest(unittest.TestCase):
    def test_01_scatter_efficiency_zero_contrast(self):
        wavelength = np.array([LAM], dtype=float)

        for name, n_match in (
            ("matched_real", 1.5),
            ("matched_complex", 1.5 + 1e-5j),
        ):
            with self.subTest(case=name):
                n_match = np.array([n_match], dtype=complex)
                qabs, qsca, gcos, an, bn = mie.scatter_efficiency(
                    wavelength=wavelength,
                    Nh=n_match,
                    Np=n_match,
                    D=DP,
                    return_coeffs=True,
                )

                np.testing.assert_allclose(qabs, [0.0], atol=1e-12)
                np.testing.assert_allclose(qsca, [0.0], atol=1e-12)
                np.testing.assert_allclose(gcos, [0.0], atol=1e-12)
                np.testing.assert_allclose(an, 0.0, atol=1e-12)
                np.testing.assert_allclose(bn, 0.0, atol=1e-12)

    def test_02_cross_section_ensemble_zero_contrast(self):
        wavelength = np.array([LAM], dtype=float)

        for name, n_match in (
            ("matched_real", 1.5),
            ("matched_complex", 1.5 + 1e-5j),
        ):
            for phase_function in (False, True):
                with self.subTest(case=name, phase_function=phase_function):
                    n_match = np.array([n_match], dtype=complex)
                    cabs, csca, gcos, phase_fun = mie.cross_section_ensemble(
                        wavelength=wavelength,
                        Nh=n_match,
                        Np=n_match,
                        D=DP,
                        fv=0.2,
                        effective_medium=False,
                        dependent_scatt=False,
                        phase_function=phase_function,
                    )

                    np.testing.assert_allclose(cabs, [0.0], atol=1e-12)
                    np.testing.assert_allclose(csca, [0.0], atol=1e-12)
                    np.testing.assert_allclose(gcos, [0.0], atol=1e-12)

                    if phase_function:
                        self.assertIsNotNone(phase_fun)
                        np.testing.assert_allclose(phase_fun.to_numpy(), 0.0, atol=1e-12)
                    else:
                        self.assertIsNone(phase_fun)


class MieAbsorbingHostTest(unittest.TestCase):
    def test_01_phase_function_matches_direct_qsca(self):
        wavelength = np.array([0.29536], dtype=float)
        n_host = np.array([0.61832 + 0.83293j], dtype=complex)
        n_particle = np.array([1.03071 + 0.00002j], dtype=complex)

        cabs_ref, csca_ref, g_ref, _ = mie.cross_section_ensemble(
            wavelength=wavelength,
            Nh=n_host,
            Np=n_particle,
            D=2.371,
            fv=0.0,
            effective_medium=False,
            dependent_scatt=False,
            phase_function=False,
        )

        cabs_pf, csca_pf, g_pf, phase_fun = mie.cross_section_ensemble(
            wavelength=wavelength,
            Nh=n_host,
            Np=n_particle,
            D=2.371,
            fv=0.0,
            effective_medium=False,
            dependent_scatt=False,
            phase_function=True,
            n_theta=721,
        )

        np.testing.assert_allclose(cabs_pf, cabs_ref, atol=1e-12)
        np.testing.assert_allclose(csca_pf, csca_ref, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(g_pf, g_ref, rtol=1e-6, atol=1e-9)
        self.assertTrue(np.all(np.isfinite(phase_fun.to_numpy())))

    def test_02_structure_factor_polydisperse_is_finite(self):
        wavelength = np.array([0.29536], dtype=float)
        theta = np.linspace(0.0, np.pi, 11)

        s_q = mie.structure_factor_PY(
            wavelength=wavelength,
            Nh=np.array([0.61832 + 0.83293j], dtype=complex),
            D=np.array([0.2887, 0.3683, 0.4609, 0.5685], dtype=float),
            fv=0.2,
            theta=theta,
            size_dist=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        )

        self.assertEqual(s_q.shape, (theta.size, wavelength.size))
        self.assertTrue(np.all(np.isfinite(s_q)))
        self.assertTrue(np.all(s_q >= 0.0))


if __name__ == "__main__":
    unittest.main()
