"""Regression tests for the Riccati-Bessel recurrence and the ensemble
angular quadrature.

Both cover changes that are invisible to the public API but easy to regress:

* ``_rho_from_DnGn`` replaced a direct ``jve``/``hankel1e`` evaluation of
  rho_n = xi_n/psi_n with a recurrence that reuses the Dn/Gn arrays. It must
  agree with the Bessel reference, including at real x near k*pi where the
  recurrence seed is singular and a fallback takes over.

* ``_ensemble_optics`` derives csca/gcos from the same Gauss-Legendre pass
  that produces the Legendre moments. The node count must scale with the size
  parameter, or forward-peaked phase functions at large x are badly
  misintegrated (the previous sqrt(x) heuristic reached 1280% error on csca).
"""
import os
import sys
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import empylib.miescattering as mie  # noqa: E402


def _build_DnGn(x, nmx):
    """Dn/Gn exactly as _log_RicattiBessel builds them, for a (nrow, 1) x."""
    D = np.zeros((x.shape[0], 1, nmx), dtype=complex)
    for i in reversed(range(1, nmx)):
        D[:, :, i - 1] = (i + 1) / x - 1 / (D[:, :, i] + (i + 1) / x)
    G = np.zeros((x.shape[0], 1, nmx), dtype=complex)
    G[:, :, 0] = 1 / (1 / x - 1j * np.ones_like(x)) - 1 / x
    for i in range(1, nmx):
        G[:, :, i] = 1 / ((i + 1) / x - G[:, :, i - 1]) - (i + 1) / x
    return D, G


class RiccatiRhoRecurrenceTest(unittest.TestCase):
    """rho = xi/psi by recurrence must match the Bessel reference."""

    def _check(self, xvals, rtol):
        x = np.asarray(xvals, dtype=complex).reshape(-1, 1)
        xa = np.abs(x).max()
        nmax = int(round(xa + 4 * xa ** (1 / 3) + 2))
        nmx = int(round(max(nmax, xa) + 16))
        Dn, Gn = _build_DnGn(x, nmx)
        got = mie._rho_from_DnGn(x, Dn, Gn, nmax)
        ref = mie._riccati_xi_over_psi(x, nmax)
        with np.errstate(all="ignore"):
            rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
        # Compare only where the reference itself is representable; at very
        # high order |rho| runs past the double-precision range.
        good = np.isfinite(ref) & (np.abs(ref) > 1e-280) & (np.abs(ref) < 1e280)
        self.assertTrue(np.any(good), "no comparable entries")
        self.assertLess(np.nanmax(rel[good]), rtol)

    def test_lossless_host(self):
        lam = np.linspace(0.30, 1.10, 61)
        self._check(2 * np.pi * 1.5 / lam * (11.7 / 2) + 0j, 1e-7)

    def test_absorbing_hosts(self):
        lam = np.linspace(0.30, 1.10, 41)
        for nh in (1.5 + 0.05j, 1.5 + 0.5j):
            with self.subTest(nh=nh):
                self._check(2 * np.pi * nh / lam * (11.7 / 2), 1e-7)

    def test_seed_singularity_at_multiples_of_pi(self):
        """x = k*pi makes psi_0 = sin(x) vanish; the fallback must take over."""
        x = np.array([k * np.pi for k in range(1, 40)], dtype=complex)
        e2 = np.exp(2j * x.reshape(-1, 1))
        self.assertTrue(
            np.all(np.abs(e2 - 1.0) < mie._RHO_SEED_FLOOR),
            "these x should all be flagged as ill-conditioned seeds",
        )
        self._check(x, 1e-10)

    def test_small_size_parameters(self):
        self._check(np.linspace(0.01, 0.9, 41) + 0j, 1e-9)


class LosslessHostSkipsPsiTest(unittest.TestCase):
    """psi_n(y) is only needed by the absorbing-host formulas."""

    def test_psi_not_evaluated_for_real_index_host(self):
        lam = np.linspace(0.4, 1.2, 21)
        calls = []
        orig = mie._riccati_psi_scaled
        mie._riccati_psi_scaled = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        try:
            mie.scatter_efficiency(lam, np.full_like(lam, 1.5, dtype=complex),
                                   np.full_like(lam, 1.2, dtype=complex), 2.0)
            self.assertEqual(calls, [], "psi should not be computed for a lossless host")
            mie.scatter_efficiency(lam, np.full_like(lam, 1.5 + 0.2j, dtype=complex),
                                   np.full_like(lam, 1.2, dtype=complex), 2.0)
            self.assertTrue(calls, "psi IS required for an absorbing host")
        finally:
            mie._riccati_psi_scaled = orig


class EnsembleQuadratureTest(unittest.TestCase):
    """csca/gcos from _ensemble_optics must be converged, at any size parameter."""

    def _reference(self, lam, nkh, nkp, D, w, fv):
        # 8 nodes per unit size parameter -- well past convergence (~1e-7),
        # so it is a sound reference for the 1e-3 assertions below without
        # making this test file slow.
        x_max = np.pi * nkh.real.max() * D.max() / lam.min()
        mu, ww = leggauss(int(max(1500, 8 * x_max)))
        o = np.argsort(mu)
        mu, ww = mu[o], ww[o]
        p = mie.phase_scatt_ensemble(lam, np.arccos(mu), nkh, nkp, D, fv, size_dist=w,
                                     mu_weights=ww, as_ndarray=True,
                                     dependent_scatt=True, effective_medium=True)
        q = 2 * np.pi * (ww @ p)
        g = 2 * np.pi * ((ww * mu) @ p) / q
        return q * float(np.sum(w * np.pi * (D / 2) ** 2)), g

    def test_converged_across_size_parameters(self):
        for Dc, Ds, lam in (
            (1.2, 0.45, np.linspace(0.25, 2.5, 40)),
            (5.0, 1.5, np.linspace(0.30, 1.10, 40)),
            (12.0, 3.5, np.linspace(0.30, 1.10, 25)),
        ):
            with self.subTest(D_mean=Dc):
                D = np.linspace(max(0.05, Dc - 3 * Ds), Dc + 3 * Ds, 7)
                w = np.exp(-(D - Dc) ** 2 / (2 * Ds ** 2))
                w /= w.sum()
                nkh = np.full_like(lam, 1.50, dtype=complex)
                nkp = np.full_like(lam, 1.05, dtype=complex)
                ref_c, ref_g = self._reference(lam, nkh, nkp, D, w, 0.30)
                _, csca, gcos, _ = mie.cross_section_ensemble(
                    lam, nkh, nkp, D, 0.30, size_dist=w, effective_medium=True,
                    dependent_scatt=True, phase_function=False)
                self.assertLess(np.nanmax(np.abs(csca - ref_c) / np.abs(ref_c)), 1e-3)
                self.assertLess(np.nanmax(np.abs(gcos - ref_g)), 1e-3)

    def test_node_count_grows_with_size_parameter(self):
        lam = np.array([0.5])
        n_small = mie._quadrature_order(lam, np.array([1.5 + 0j]), np.array([1.0]),
                                        n_moments=33)
        n_large = mie._quadrature_order(lam, np.array([1.5 + 0j]), np.array([20.0]),
                                        n_moments=33)
        self.assertGreater(n_large, 4 * n_small)
        # never below what the requested moments need
        self.assertGreaterEqual(n_small, 4 * 33)

    def test_single_angular_pass(self):
        """adm_sphere must evaluate the phase function and S(q) exactly once."""
        import empylib.dense_spheres as ds
        import empylib.rad_transfer as rt

        counts = {"pf": 0, "sq": 0}
        o_pf, o_sq = mie.phase_scatt_ensemble, mie.structure_factor_PY
        mie.phase_scatt_ensemble = lambda *a, **k: (counts.__setitem__("pf", counts["pf"] + 1),
                                                    o_pf(*a, **k))[1]
        ds.structure_factor_PY = lambda *a, **k: (counts.__setitem__("sq", counts["sq"] + 1),
                                                  o_sq(*a, **k))[1]
        mie.structure_factor_PY = ds.structure_factor_PY
        try:
            lam = np.linspace(0.4, 1.0, 12)
            D = np.linspace(0.2, 2.0, 5)
            w = np.ones_like(D) / D.size
            rt.adm_sphere(wavelength=lam,
                          N_host=np.full_like(lam, 1.5, dtype=complex),
                          N_particle=np.full_like(lam, 1.1, dtype=complex),
                          D=D, fv=0.2, thickness=0.1, size_dist=w,
                          effective_medium=True, dependent_scatt=True,
                          use_phase_fun=True)
        finally:
            mie.phase_scatt_ensemble = o_pf
            mie.structure_factor_PY = o_sq
            ds.structure_factor_PY = o_sq
        self.assertEqual(counts["pf"], 1, "phase function evaluated more than once")
        self.assertEqual(counts["sq"], 1, "structure factor evaluated more than once")


if __name__ == "__main__":
    unittest.main(verbosity=1)
