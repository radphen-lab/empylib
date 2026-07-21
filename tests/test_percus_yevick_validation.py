import os
import unittest
import warnings

import numpy as np
from scipy.special import gammaln

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import empylib.miescattering as mie
import empylib.dense_spheres as ds


# ---------------------------------------------------------------------------
# Test-local Schulz closed-form helper (Botet, Kwok & Cabane 2020, Eqs. 4-6,
# 34-47), written independently of empylib.dense_spheres.SchulzDistribution
# so comparisons against it are not tautological.
# ---------------------------------------------------------------------------
def _botet_schulz_closed_form(fv, q, D_mean, s):
    a_mean = D_mean / 2.0
    psi = 3.0 * fv / (1.0 - fv)
    x = 2.0 * q * a_mean / s                                    # Eq. 36
    theta = (s + 2) * np.arctan(x)
    R = np.cos(theta) / (1 + x**2)**(1 + s/2.0)                 # Eq. 37
    I = np.sin(theta) / (1 + x**2)**(1 + s/2.0)                 # Eq. 38
    g1 = psi / ((s + 2) * x)                                     # Eq. 39
    g2 = 2*psi / ((s + 1)*(s + 2)*x**2)                          # Eq. 40
    g3 = 4*psi / (s*(s + 1)*(s + 2)*x**3)                        # Eq. 41
    b = (g3 - (s+4)/s*g1)*I - 2*(s+2)/s*g2*R                     # Eq. 42
    c = g1 + g3 - (g3 - (s+4)/s*g1)*R - 2*(s+2)/s*g2*I           # Eq. 43
    d = 1 + g1*I                                                 # Eq. 44
    e = g1 - g1*R                                                # Eq. 45
    f = g2 - g2*R - (s+3)/(s+1)*g1*I                             # Eq. 46
    g = g1 - g2*I + (s+3)/(s+1)*g1*R                             # Eq. 47
    denom = d**2 + e**2
    X = 1 + b + (2*e*f*g + d*(f**2 - g**2)) / denom              # Eq. 5
    Y = c + (2*d*f*g - e*(f**2 - g**2)) / denom                  # Eq. 6
    return (Y / c) / (X**2 + Y**2)                               # Eq. 4


# ---------------------------------------------------------------------------
# Test-local Mitsuaki (Ginoza & Yasutomi 1999) single-species Schulz-diameter
# oracle -- independent literature check only, never shipped in empylib.
# Eqs. 15, 18, 26-29. If a test using this fails, first re-verify the
# transcribed equations against the original paper figures before assuming
# a bug in empylib.dense_spheres.
# ---------------------------------------------------------------------------
def _mitsuaki_t_m(t, m):
    """Eq. 24: t_m = (t+m)! / (t! (t+1)**m), via gammaln for large-t safety."""
    return np.exp(gammaln(t + m + 1) - gammaln(t + 1)) / (t + 1)**m


def _mitsuaki_f_m(t, m, a):
    """Eq. 25: exponentially-weighted moment f_m^tau(a)."""
    return _mitsuaki_t_m(t, m) * (1 + a/(t+1))**(-(t + m + 1))


def _mitsuaki_single_species_SM(k, eta, sigma0, t):
    s = 1j * k
    t2 = _mitsuaki_t_m(t, 2)
    t3 = _mitsuaki_t_m(t, 3)
    rho = eta / ((np.pi/6.0) * sigma0**3 * t3)
    Delta = 1.0 - eta
    zeta2 = rho * sigma0**2 * t2

    x = s * sigma0
    f0p, f1p, f2p = _mitsuaki_f_m(t, 0, x), _mitsuaki_f_m(t, 1, x), _mitsuaki_f_m(t, 2, x)
    f0m, f1m, f2m = _mitsuaki_f_m(t, 0, -x), _mitsuaki_f_m(t, 1, -x), _mitsuaki_f_m(t, 2, -x)

    # Eqs. 26a/26b (single species: F0 = 1, c = 1)
    Ialpha1 = (24.0/s**3) * (-0.5*(1 - f0p) + s*sigma0/4.0*(1 + f1p))
    Ialpha2 = (24.0/s**3) * (-(sigma0**2)/2.0*(1 - f1p) + s*sigma0**2/4.0*(t2 + f2p))

    # Eqs. 27a/27b
    Iw1 = (2*np.pi*rho/(Delta*s**3)) * (Ialpha1 + s/2.0*Ialpha2)
    Iw2 = ((np.pi*rho/(Delta*s**2)) * (1 + np.pi*zeta2/(Delta*s)) * Ialpha1
           + (np.pi**2*zeta2*rho/(2*Delta**2*s**2)) * Ialpha2)

    # Eq. 27c
    I0 = -(9.0/2)*(2.0/s)**6 * (1 - 0.5*(f0m + f0p) + s*sigma0/2.0*(f1m - f1p)
                                - s**2*sigma0**2/8.0*(f2m + f2p + 2*t2))

    # Eqs. 29a-d
    fa = (1 - x/2.0 - f0p - x*f1p/2.0) / x**3
    fb = (1 - x*t2/2.0 - f1p - x*f2p/2.0) / x**3
    fc = (1 - x - f0p) / x**2
    fd = (1 - x*t2 - f1p) / x**2

    # Eqs. 28a-d (single species)
    F11 = (2*np.pi*rho*sigma0**3/Delta) * fa
    F21 = sigma0 * (2*np.pi*rho*sigma0**3/Delta) * fb
    F12 = (1.0/sigma0) * ((np.pi/Delta)**2*rho*zeta2*sigma0**4*fa + np.pi*rho*sigma0**3/Delta*fc)
    F22 = (np.pi/Delta)**2*rho*zeta2*sigma0**4*fb + np.pi*rho*sigma0**3/Delta*fd

    Mmat = np.array([[1 - F11, -F12], [-F21, 1 - F22]])
    Ghat = np.linalg.inv(Mmat)
    Iw = np.array([Iw1, Iw2])
    Ialpha = np.array([Ialpha1, Ialpha2])
    return 1 - 2*np.real(Iw @ Ghat @ Ialpha / I0)


class MonoDiluteLargeQLimitsTest(unittest.TestCase):

    def test_01_mono_limit_matches_poly_near_delta(self):
        D0 = 0.30
        D = np.array([D0*(1 - 1e-4), D0, D0*(1 + 1e-4)], dtype=float)
        nD = np.array([1.0, 1.0, 1.0], dtype=float)
        q = np.linspace(1.0, 40.0, 25)[None, :]

        for fv in (0.1, 0.3, 0.45):
            with self.subTest(fv=fv):
                S_poly = ds._poly_percus_yevick(fv, q, D, nD)
                S_mono = ds._mono_percus_yevick(fv, q, D0)
                np.testing.assert_allclose(S_poly, S_mono, rtol=1e-3)

    def test_02_dilute_limit_S_to_1(self):
        fv = 1e-6
        q = np.linspace(0.5, 50.0, 30)[None, :]

        D_uniform = np.linspace(0.2, 0.4, 9)
        nD_uniform = np.ones_like(D_uniform)
        S_uniform = ds._poly_percus_yevick(fv, q, D_uniform, nD_uniform)
        np.testing.assert_allclose(S_uniform, 1.0, atol=1e-3)

        D_exp = np.linspace(1e-5, 3.0, 100)
        nD_exp = np.exp(-D_exp/0.3)
        S_exp = ds._poly_percus_yevick(fv, q, D_exp, nD_exp)
        np.testing.assert_allclose(S_exp, 1.0, atol=1e-3)

        # end-to-end via the public dispatcher
        s_q = mie.structure_factor_PY(
            wavelength=np.array([0.5]),
            Nh=np.array([1.5 + 0j]),
            D=D_uniform,
            fv=fv,
            theta=np.linspace(0.1, np.pi, 20),
            size_dist=nD_uniform,
        )
        np.testing.assert_allclose(s_q, 1.0, atol=1e-3)

    def test_03_large_q_limit_S_to_1_from_above(self):
        fv = 0.3
        D_mean = 0.3
        D = np.linspace(0.2, 0.4, 9)
        nD = np.ones_like(D)
        q = np.linspace(1.0, 200.0/D_mean, 400)[None, :]

        S_q = ds._poly_percus_yevick(fv, q, D, nD)[0]
        tail = S_q[-20:]
        self.assertTrue(np.all(tail >= 1.0 - 1e-6))

        K_estimate = (S_q - 1.0) * q[0]**2
        k1, k2 = K_estimate[-40], K_estimate[-1]
        self.assertAlmostEqual(k1, k2, delta=0.05*max(abs(k1), abs(k2), 1e-12))

    def test_07_small_q_smoothness(self):
        fv = 0.3
        D = np.linspace(0.2, 0.4, 9)
        nD = np.ones_like(D)
        q = np.linspace(1e-6, 1.0, 500)[None, :]

        S_q = ds._poly_percus_yevick(fv, q, D, nD)[0]
        self.assertTrue(np.all(np.isfinite(S_q)))
        second_diff = np.diff(S_q, 2)
        self.assertTrue(np.all(np.abs(second_diff) < 1.0))


class ComplementaryPYWarningTest(unittest.TestCase):

    def test_08_complementary_py_emits_warning(self):
        D = np.linspace(0.2, 0.4, 5)
        nD = np.ones_like(D)
        q = np.linspace(1.0, 20.0, 10)[None, :]
        with self.assertWarns(UserWarning):
            ds._poly_percus_yevick(0.6, q, D, nD)


class SchulzClosedFormVsTrapzTest(unittest.TestCase):

    def test_04_schulz_closed_form_vs_trapz(self):
        D_mean = 0.3
        fv = 0.25
        q = np.linspace(1.0, 60.0, 40)[None, :]

        for s in (2, 5, 10):
            with self.subTest(s=s):
                D_grid = np.linspace(1e-3, 10*D_mean, 4000)
                nD_grid = D_grid**(s - 1) * np.exp(-s*D_grid/D_mean)

                S_trapz = ds._poly_percus_yevick(fv, q, D_grid, nD_grid)
                S_closed_local = _botet_schulz_closed_form(fv, q, D_mean, float(s))
                S_closed_prod = ds.schulz(D_mean, s)._closed_form_S(fv, q)

                np.testing.assert_allclose(S_trapz, S_closed_local, rtol=1e-3)
                np.testing.assert_allclose(S_closed_local, S_closed_prod, rtol=1e-8)


class OtherClosedFormVsTrapzTest(unittest.TestCase):
    """Cross-checks the three closed-form kernels that have no independent
    literature oracle (unlike Schulz) against empylib's own trapz-integrated
    kernel, fed a fine tabulation built from each distribution's own pdf()."""

    def test_09_truncated_normal_closed_form_vs_trapz(self):
        D_mean, p, fv = 0.3, 0.05, 0.2
        q = np.linspace(1.0, 60.0, 40)[None, :]
        dist = ds.truncated_normal(D_mean, p)
        std_D = D_mean * np.sqrt(p)
        D_grid = np.linspace(max(1e-4, D_mean - 8*std_D), D_mean + 8*std_D, 4000)
        nD_grid = dist.pdf(D_grid)

        S_trapz = ds._poly_percus_yevick(fv, q, D_grid, nD_grid)
        S_closed = dist._closed_form_S(fv, q)
        np.testing.assert_allclose(S_trapz, S_closed, rtol=2e-3)

    def test_10_inverse_gaussian_closed_form_vs_trapz(self):
        D_mean, p, fv = 0.3, 0.05, 0.2
        q = np.linspace(1.0, 60.0, 40)[None, :]
        dist = ds.inverse_gaussian(D_mean, p)
        std_D = D_mean * np.sqrt(p)
        D_grid = np.linspace(max(1e-4, D_mean - 8*std_D), D_mean + 8*std_D, 4000)
        nD_grid = dist.pdf(D_grid)

        S_trapz = ds._poly_percus_yevick(fv, q, D_grid, nD_grid)
        S_closed = dist._closed_form_S(fv, q)
        np.testing.assert_allclose(S_trapz, S_closed, rtol=2e-3)

    def test_11_exponential_closed_form_vs_trapz(self):
        D_mean, fv = 0.3, 0.2
        q = np.linspace(1.0, 60.0, 40)[None, :]
        dist = ds.exponential(D_mean)
        D_grid = np.linspace(1e-4, 15*D_mean, 6000)
        nD_grid = dist.pdf(D_grid)

        S_trapz = ds._poly_percus_yevick(fv, q, D_grid, nD_grid)
        S_closed = dist._closed_form_S(fv, q)
        np.testing.assert_allclose(S_trapz, S_closed, rtol=3e-3)


class MitsuakiOracleTest(unittest.TestCase):
    """Independent oracle only (Ginoza & Yasutomi 1999); not shipped in
    empylib. If this fails, re-verify the transcribed Mitsuaki equations
    against the original paper figures before assuming a bug in
    empylib.dense_spheres."""

    def test_05_mitsuaki_vs_botet_schulz(self):
        eta = 0.3
        sigma0 = 1.0
        for D_sigma in (0.5, 0.2, 0.1):
            with self.subTest(D_sigma=D_sigma):
                t = round(1.0/D_sigma - 1.0)
                self.assertAlmostEqual(1.0/D_sigma - 1.0, t, places=6)
                s_botet = t + 1  # Mitsuaki's t and Botet's s: D_sigma == p == 1/s

                k_grid = np.linspace(0.1, 15.0, 30) / sigma0
                S_botet = _botet_schulz_closed_form(eta, k_grid[None, :], sigma0, float(s_botet))[0]
                S_mitsuaki = np.array([
                    _mitsuaki_single_species_SM(k, eta, sigma0, t) for k in k_grid
                ])
                np.testing.assert_allclose(S_botet, S_mitsuaki, rtol=1e-2)


class EnsembleDependentScattTest(unittest.TestCase):

    def _tabulated_inputs(self):
        return dict(
            wavelength=np.array([0.29536], dtype=float),
            Nh=np.array([0.61832 + 0.83293j], dtype=complex),
            Np=1.5 + 0.01j,
            D=np.array([0.2887, 0.3683, 0.4609, 0.5685], dtype=float),
            fv=0.2,
            size_dist=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        )

    def test_06_phase_scatt_ensemble_dependent_scatt_polydisperse(self):
        kw = self._tabulated_inputs()
        theta = np.linspace(0.0, np.pi, 11)

        phase_dep = mie.phase_scatt_ensemble(
            kw["wavelength"], theta, kw["Nh"], kw["Np"], kw["D"], kw["fv"],
            size_dist=kw["size_dist"], as_ndarray=True, dependent_scatt=True,
        )
        phase_indep = mie.phase_scatt_ensemble(
            kw["wavelength"], theta, kw["Nh"], kw["Np"], kw["D"], kw["fv"],
            size_dist=kw["size_dist"], as_ndarray=True, dependent_scatt=False,
        )
        self.assertTrue(np.all(np.isfinite(phase_dep)))
        self.assertTrue(np.all(phase_dep >= 0.0))
        self.assertFalse(np.allclose(phase_dep, phase_indep))

        cabs_d, csca_d, g_d, phase_df_d = mie.cross_section_ensemble(
            kw["wavelength"], kw["Nh"], kw["Np"], kw["D"], kw["fv"],
            size_dist=kw["size_dist"], dependent_scatt=True, phase_function=True,
        )
        cabs_i, csca_i, g_i, phase_df_i = mie.cross_section_ensemble(
            kw["wavelength"], kw["Nh"], kw["Np"], kw["D"], kw["fv"],
            size_dist=kw["size_dist"], dependent_scatt=False, phase_function=True,
        )
        self.assertTrue(np.all(np.isfinite(csca_d)) and np.all(csca_d >= 0.0))
        self.assertFalse(np.allclose(phase_df_d.to_numpy(), phase_df_i.to_numpy()))


class RegressionPinTest(unittest.TestCase):
    """Pins structure_factor_PY's output for ndarray size_dist so the
    dense_spheres.py refactor cannot silently change tabulated-path
    behavior."""

    def test_12_ndarray_size_dist_output_unchanged(self):
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
        expected = np.array([
            [0.22301417], [0.30352421], [0.68741571], [1.17938184],
            [1.07756176], [1.02713851], [1.03058689], [1.03368880],
            [1.02815325], [1.02409980], [1.02285677],
        ])
        np.testing.assert_allclose(s_q, expected, rtol=1e-6, atol=1e-8)


class ClosedFormDistributionEndToEndTest(unittest.TestCase):
    """Verifies the closed-form distribution API works end to end through
    all three public entry points, with D=None, and that the analytic path
    is actually reached (not silently falling back to trapz)."""

    def test_13_structure_factor_PY_with_D_none(self):
        s_q = mie.structure_factor_PY(
            wavelength=np.array([0.5]), Nh=np.array([1.5 + 0j]), D=None, fv=0.2,
            theta=np.linspace(0.1, np.pi, 15), size_dist=ds.schulz(0.3, 5),
        )
        self.assertTrue(np.all(np.isfinite(s_q)))
        self.assertTrue(np.all(s_q >= 0.0))

    def test_14_phase_scatt_ensemble_with_D_none(self):
        phase = mie.phase_scatt_ensemble(
            np.array([0.5]), np.linspace(0.1, np.pi, 15), np.array([1.5 + 0j]),
            1.5 + 0.01j, None, 0.2,
            size_dist=ds.schulz(0.3, 5), as_ndarray=True, dependent_scatt=True,
        )
        self.assertTrue(np.all(np.isfinite(phase)))
        self.assertTrue(np.all(phase >= 0.0))

    def test_15_cross_section_ensemble_with_D_none(self):
        cabs, csca, g, _ = mie.cross_section_ensemble(
            np.array([0.5]), np.array([1.5 + 0j]), 1.5 + 0.01j, None, 0.2,
            size_dist=ds.schulz(0.3, 5), dependent_scatt=True, phase_function=False,
        )
        self.assertTrue(np.all(np.isfinite(csca)) and np.all(csca >= 0.0))

    def test_16_cross_section_ensemble_reaches_analytic_path(self):
        """cross_section_ensemble(dependent_scatt=True, phase_function=True)
        with a closed-form distribution must reach structure_factor_PY's
        analytic kernel, not silently fall back to the trapz kernel, even
        though it internally forwards through phase_scatt_ensemble."""
        import unittest.mock as mock

        with mock.patch.object(ds, "_poly_percus_yevick", wraps=ds._poly_percus_yevick) as spy:
            cabs, csca, g, phase_df = mie.cross_section_ensemble(
                np.array([0.5]), np.array([1.5 + 0j]), 1.5 + 0.01j, None, 0.2,
                size_dist=ds.schulz(0.3, 5), dependent_scatt=True, phase_function=True,
                n_theta=21,
            )
            spy.assert_not_called()
        self.assertTrue(np.all(np.isfinite(csca)))


if __name__ == "__main__":
    unittest.main()
