"""Regression coverage for SciPy's keyword-only Simpson coordinates."""

import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scipy.integrate import simpson as scipy_simpson

import empylib.miescattering as mie


class TestScipyIntegrationCompatibility(unittest.TestCase):
    @staticmethod
    def _keyword_only_backend(calls):
        def keyword_only_simpson(y, *, x, axis=-1):
            calls.append(np.asarray(x))
            return scipy_simpson(y, x=x, axis=axis)

        return keyword_only_simpson

    def test_scatter_from_phase_function_accepts_keyword_only_simpson_x(self):
        calls = []
        theta = np.linspace(0.0, 180.0, 101)
        phase = pd.DataFrame({0.5: np.ones(theta.size)}, index=theta)

        with mock.patch.object(mie, "_simpson", self._keyword_only_backend(calls)):
            qsca, gcos = mie.scatter_from_phase_function(phase)

        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(qsca, [4.0 * np.pi], rtol=1e-12)
        np.testing.assert_allclose(gcos, [0.0], atol=1e-14)

    def test_phase_scatt_ensemble_accepts_keyword_only_simpson_x(self):
        calls = []
        theta = np.linspace(0.0, np.pi, 101)

        with mock.patch.object(mie, "_simpson", self._keyword_only_backend(calls)):
            phase = mie.phase_scatt_ensemble(
                0.5, theta, 1.0, 1.5 + 0.02j, 0.6, as_ndarray=True
            )

        self.assertEqual(len(calls), 1)
        self.assertTrue(np.all(np.isfinite(phase)))
