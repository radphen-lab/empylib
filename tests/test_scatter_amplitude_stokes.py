import os
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import empylib.miescattering as mie

try:
    import miepython
    _HAS_MIEPYTHON = True
except ImportError:
    _HAS_MIEPYTHON = False


class ScatterAmplitudeStokesConsistencyTest(unittest.TestCase):
    """Golden-value / self-consistency checks for scatter_amplitude and
    scatter_stokes, added while auditing miescattering.py for a S34 formula
    bug (was `1*2*(...)`, giving a purely-imaginary, wrongly-scaled result
    instead of the real Bohren & Huffman S34 = Im(i/2*(S1 S2* - S2 S1*)))."""

    CASES = [
        # (Nh, Np, D) -- homogeneous sphere, non-trivial size parameter.
        (1.0, 1.5 + 0.0j, 0.6),
        (1.0, 1.5 + 0.1j, 0.6),   # absorbing particle
        (1.33, 2.0 + 0.05j, 0.3),
    ]

    def _s1_s2(self, Nh, Np, D, theta):
        s1, s2 = mie.scatter_amplitude(
            np.array([0.5]), Nh, Np, D, theta=theta, check_inputs=True
        )
        return s1[:, 0], s2[:, 0]

    def test_S11_S12_match_amplitude_combination(self):
        theta = np.linspace(0.05, np.pi - 0.05, 11)
        for Nh, Np, D in self.CASES:
            s1, s2 = self._s1_s2(Nh, Np, D, theta)
            S11, S12, S33, S34 = mie.scatter_stokes(
                np.array([0.5]), Nh, Np, D, theta=theta, check_inputs=True
            )
            S11 = S11[:, 0]; S12 = S12[:, 0]

            np.testing.assert_allclose(
                S11.real, 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2), rtol=1e-10
            )
            np.testing.assert_allclose(
                S12.real, 0.5 * (np.abs(s1) ** 2 - np.abs(s2) ** 2), rtol=1e-10
            )

    def test_S33_S34_are_real_and_match_bohren_huffman_convention(self):
        """S33 = Re(S1 conj(S2)); S34 = -Im(S1 conj(S2)) (Bohren & Huffman
        1983, eq. 3.16, S34 = (i/2)(S1 S2* - S2 S1*)). Before the fix, S34
        came out purely imaginary with 4x the correct magnitude."""
        theta = np.linspace(0.05, np.pi - 0.05, 11)
        for Nh, Np, D in self.CASES:
            s1, s2 = self._s1_s2(Nh, Np, D, theta)
            S11, S12, S33, S34 = mie.scatter_stokes(
                np.array([0.5]), Nh, Np, D, theta=theta, check_inputs=True
            )
            S33 = S33[:, 0]; S34 = S34[:, 0]

            expected_S33 = np.real(s1 * np.conj(s2))
            expected_S34 = -np.imag(s1 * np.conj(s2))

            # S34 must be (numerically) real -- the pre-fix bug made it
            # purely imaginary.
            np.testing.assert_allclose(np.imag(S34), 0.0, atol=1e-8)

            np.testing.assert_allclose(S33.real, expected_S33, rtol=1e-10)
            np.testing.assert_allclose(S34.real, expected_S34, rtol=1e-10)

    @unittest.skipUnless(_HAS_MIEPYTHON, "miepython not installed")
    def test_S34_over_S33_ratio_matches_miepython_in_magnitude(self):
        """Normalization-independent cross-check against an independent Mie
        implementation: |S34/S33| depends only on the relative phase/
        magnitude of S1 and S2 at each angle, which is invariant to the
        overall (real, angle-independent) amplitude normalization used by
        either package."""
        lam = 0.5
        theta = np.linspace(0.2, np.pi - 0.2, 9)
        mu = np.cos(theta)
        for Nh, Np, D in self.CASES:
            s1, s2 = self._s1_s2(Nh, Np, D, theta)
            _, _, S33, S34 = mie.scatter_stokes(
                np.array([lam]), Nh, Np, D, theta=theta, check_inputs=True
            )
            ours_ratio = np.abs(S34[:, 0].real / S33[:, 0].real)

            x = np.pi * Nh * D / lam
            m = Np / Nh
            S1m, S2m = miepython.S1_S2(m, x, mu, norm="wiscombe")
            ref = S1m * np.conj(S2m)
            mp_ratio = np.abs(np.imag(ref) / np.real(ref))

            np.testing.assert_allclose(ours_ratio, mp_ratio, rtol=1e-6, atol=1e-9)


class ScatterFromPhaseFunctionGoldenTest(unittest.TestCase):
    """Cross-checks scatter_from_phase_function's Simpson-over-mu integration
    against the independently-implemented, analytic phase_scatt_HG, whose
    Qsca and <cos theta> are known exactly by construction."""

    def test_recovers_hg_qsca_and_gcos(self):
        wavelength = np.array([0.5])
        g = 0.6
        theta = np.linspace(0.0, np.pi, 721)

        p_hg = mie.phase_scatt_HG(wavelength, gcos=g, qsca=1.0, theta=theta)
        qsca, gcos = mie.scatter_from_phase_function(p_hg)

        np.testing.assert_allclose(qsca, [1.0], atol=1e-6)
        np.testing.assert_allclose(gcos, [g], atol=1e-6)

    def test_recovers_hg_qsca_and_gcos_degrees_index(self):
        wavelength = np.array([0.5])
        g = -0.4
        theta_deg = np.linspace(0.0, 180.0, 721)

        p_hg = mie.phase_scatt_HG(
            wavelength, gcos=g, qsca=2.0, theta=np.radians(theta_deg)
        )
        p_hg.index = theta_deg
        p_hg.index.name = "Theta (deg)"
        qsca, gcos = mie.scatter_from_phase_function(p_hg)

        np.testing.assert_allclose(qsca, [2.0], atol=1e-5)
        np.testing.assert_allclose(gcos, [g], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
