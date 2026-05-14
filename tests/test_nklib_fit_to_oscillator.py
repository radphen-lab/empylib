import os
import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from empylib import nklib as nk


def _fit_df(wavelength, **columns):
    df = pd.DataFrame(columns, index=wavelength)
    df.index.name = "wavelength"
    return df


class _FakeSlider:
    def __init__(self, value, min, max, step, description, continuous_update=True):
        self._value = value
        self.min = min
        self.max = max
        self.step = step
        self.description = description
        self.continuous_update = continuous_update
        self.closed = False
        self._observers = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        old = self._value
        self._value = value
        for observer, names in list(self._observers):
            observer({"name": names, "old": old, "new": value, "owner": self})

    def observe(self, observer, names="value"):
        self._observers.append((observer, names))

    def unobserve(self, observer, names="value"):
        self._observers = [
            item for item in self._observers
            if item != (observer, names)
        ]

    def close(self):
        self.closed = True


class _FakeVBox:
    def __init__(self, children):
        self.children = list(children)


class _FakeHBox:
    def __init__(self, children):
        self.children = list(children)


class _FakeLabel:
    def __init__(self, value):
        self.value = value


class _FakeObserver:
    def __init__(self, func, sliders):
        self.func = func
        self.sliders = sliders
        self.closed = False
        self.refresh()

    def refresh(self):
        self.func(**{name: slider.value for name, slider in self.sliders.items()})

    def close(self):
        self.closed = True


class _FakeOutput:
    def __init__(self):
        self.closed = False
        self.enter_count = 0
        self.exit_count = 0
        self.clear_output_calls = []

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_count += 1
        return False

    def close(self):
        self.closed = True

    def clear_output(self, *args, **kwargs):
        self.clear_output_calls.append((args, kwargs))


class _FakeAxes:
    def __init__(self):
        self.plot_calls = []
        self.ylabel = None
        self.xlabel = None
        self.ylim = None
        self.xlim = None
        self.lines = []
        self.legend_calls = 0
        self._color_idx = 0
        self.yscale = "linear"

    def plot(self, *args, **kwargs):
        color = kwargs.get("color")
        if color is None:
            color = f"C{self._color_idx}"
            self._color_idx += 1
        self.plot_calls.append((args, kwargs))
        ydata = args[1] if len(args) > 1 else None
        line = _FakeLine(color, ydata)
        self.lines.append(line)
        return [line]

    def set_ylabel(self, *args, **kwargs):
        self.ylabel = args[0] if args else None
        return None

    def set_xlabel(self, *args, **kwargs):
        self.xlabel = args[0] if args else None
        return None

    def set_ylim(self, *args, **kwargs):
        self.ylim = args
        return None

    def get_ylim(self):
        return self.ylim

    def set_xlim(self, *args, **kwargs):
        self.xlim = args
        return None

    def set_yscale(self, scale):
        self.yscale = scale
        return None

    def get_yscale(self):
        return self.yscale

    def legend(self, *args, **kwargs):
        self.legend_calls += 1
        return None


class _FakeLine:
    def __init__(self, color, ydata=None):
        self._color = color
        self.ydata = ydata

    def get_color(self):
        return self._color

    def set_ydata(self, ydata):
        self.ydata = ydata
        return None


class _FakeCanvas:
    def __init__(self):
        self.draw_idle_calls = 0

    def draw_idle(self):
        self.draw_idle_calls += 1


class _FakeFigure:
    def __init__(self):
        self.canvas = _FakeCanvas()
        self.title = None
        self.tight_layout_calls = 0

    def suptitle(self, *args, **kwargs):
        self.title = args[0] if args else None
        return None

    def tight_layout(self):
        self.tight_layout_calls += 1
        return None


def _make_fake_widget_modules():
    widgets_mod = types.ModuleType("ipywidgets")
    widgets_mod.VBox = _FakeVBox
    widgets_mod.HBox = _FakeHBox
    widgets_mod.Label = _FakeLabel
    widgets_mod.FloatSlider = _FakeSlider
    widgets_mod.Output = _FakeOutput
    widgets_mod.interactive_output = lambda func, sliders: _FakeObserver(func, sliders)

    pyplot_mod = types.ModuleType("matplotlib.pyplot")
    pyplot_mod.subplot_calls = []
    pyplot_mod.created_axes = []
    pyplot_mod.closed_figures = []
    pyplot_mod.close = lambda *args, **kwargs: pyplot_mod.closed_figures.append((args, kwargs))
    pyplot_mod.show = lambda *args, **kwargs: None
    pyplot_mod.display_calls = []

    def _display(obj):
        pyplot_mod.display_calls.append(obj)

    display_mod = types.ModuleType("IPython.display")
    display_mod.display = _display
    ipython_mod = types.ModuleType("IPython")
    ipython_mod.display = display_mod
    ipython_mod.get_ipython = lambda: object()
    pyplot_mod.ipython_mod = ipython_mod
    pyplot_mod.display_mod = display_mod

    def _subplots(nrows, ncols, squeeze=False, **kwargs):
        axes = np.array([[_FakeAxes() for _ in range(ncols)] for _ in range(nrows)], dtype=object)
        pyplot_mod.subplot_calls.append((nrows, ncols, squeeze, kwargs))
        pyplot_mod.created_axes.extend(axes.ravel().tolist())
        return _FakeFigure(), axes

    pyplot_mod.subplots = _subplots
    matplotlib_mod = types.ModuleType("matplotlib")
    matplotlib_mod.pyplot = pyplot_mod
    return widgets_mod, matplotlib_mod, pyplot_mod


def _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod):
    return {
        "ipywidgets": widgets_mod,
        "matplotlib": matplotlib_mod,
        "matplotlib.pyplot": pyplot_mod,
        "IPython": pyplot_mod.ipython_mod,
        "IPython.display": pyplot_mod.display_mod,
    }


class FitToOscillatorTest(unittest.TestCase):
    def test_fit_uses_single_least_squares_from_initial_guess(self):
        lam = np.linspace(0.6, 1.2, 40)
        target = nk.multi_oscillator(
            lam,
            {"lorentz_1": {"type": "lorentz", "epsinf": 1.0, "wp": 4.0, "wn": 2.2, "gamma": 0.18}},
        )
        guess = {
            "lorentz_1": {"type": "lorentz", "epsinf": 3.0, "wp": 1.0, "wn": 5.0, "gamma": 0.9},
        }
        expected_p0 = np.array([3.0, 1.0, 5.0, 0.9])
        y_data = _fit_df(lam, n=target.real, k=target.imag)

        with mock.patch("empylib.nklib._least_squares", wraps=nk._least_squares) as ls_mock:
            fit, res = nk.fit_to_oscillator(lam, y_data, guess)

        self.assertEqual(ls_mock.call_count, 1)
        np.testing.assert_allclose(ls_mock.call_args.args[1], expected_p0, rtol=0, atol=0)
        self.assertNotIn("method", ls_mock.call_args.kwargs)
        self.assertEqual(ls_mock.call_args.kwargs["verbose"], 0)
        self.assertLess(res.cost, 1e-12)
        np.testing.assert_allclose(
            [fit.model["lorentz_1"][name] for name in ("epsinf", "wp", "wn", "gamma")],
            [1.0, 4.0, 2.2, 0.18],
            rtol=1e-6,
            atol=1e-8,
        )

    def test_least_squares_args_are_forwarded(self):
        lam = np.linspace(0.6, 1.2, 40)
        target = nk.multi_oscillator(
            lam,
            {"lorentz_1": {"type": "lorentz", "epsinf": 1.0, "wp": 4.0, "wn": 2.2, "gamma": 0.18}},
        )
        guess = {
            "lorentz_1": {"type": "lorentz", "epsinf": 3.0, "wp": 1.0, "wn": 5.0, "gamma": 0.9},
        }
        y_data = _fit_df(lam, n=target.real, k=target.imag)
        fake_res = types.SimpleNamespace(x=np.array([3.0, 1.0, 5.0, 0.9]), cost=0.0, success=True)

        with mock.patch("empylib.nklib._least_squares", return_value=fake_res) as ls_mock:
            nk.fit_to_oscillator(
                lam,
                y_data,
                guess,
                least_squares_args={"method": "dogbox", "max_nfev": 5, "loss": "linear"},
            )

        self.assertEqual(ls_mock.call_args.kwargs["method"], "dogbox")
        self.assertEqual(ls_mock.call_args.kwargs["max_nfev"], 5)
        self.assertEqual(ls_mock.call_args.kwargs["loss"], "linear")
        self.assertEqual(ls_mock.call_args.kwargs["verbose"], 0)

    def test_verbose_is_forwarded_separately_from_least_squares_args(self):
        lam = np.linspace(0.6, 1.2, 40)
        target = nk.multi_oscillator(
            lam,
            {"lorentz_1": {"type": "lorentz", "epsinf": 1.0, "wp": 4.0, "wn": 2.2, "gamma": 0.18}},
        )
        guess = {
            "lorentz_1": {"type": "lorentz", "epsinf": 3.0, "wp": 1.0, "wn": 5.0, "gamma": 0.9},
        }
        y_data = _fit_df(lam, n=target.real, k=target.imag)
        fake_res = types.SimpleNamespace(x=np.array([3.0, 1.0, 5.0, 0.9]), cost=0.0, success=True)

        with mock.patch("empylib.nklib._least_squares", return_value=fake_res) as ls_mock:
            nk.fit_to_oscillator(lam, y_data, guess, verbose=1)

        self.assertEqual(ls_mock.call_args.kwargs["verbose"], 1)

    def test_least_squares_args_must_be_dict(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam, n=np.ones_like(lam), k=np.zeros_like(lam))

        with self.assertRaisesRegex(TypeError, "least_squares_args must be a dict"):
            nk.fit_to_oscillator(lam, y_data, oscillator, least_squares_args=[("method", "trf")])

    def test_least_squares_args_reject_reserved_keys(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam, n=np.ones_like(lam), k=np.zeros_like(lam))

        with self.assertRaisesRegex(ValueError, "controlled by fit_to_oscillator"):
            nk.fit_to_oscillator(lam, y_data, oscillator, least_squares_args={"bounds": (0, 1)})
        with self.assertRaisesRegex(ValueError, "controlled by fit_to_oscillator"):
            nk.fit_to_oscillator(lam, y_data, oscillator, least_squares_args={"verbose": 1})

    def test_direct_mode_fits_multiple_nk_samples(self):
        lam = np.linspace(0.6, 1.2, 40)
        target = nk.multi_oscillator(
            lam,
            {"lorentz_1": {"type": "lorentz", "epsinf": 1.0, "wp": 4.0, "wn": 2.2, "gamma": 0.18}},
        )
        guess = {
            "lorentz_1": {"type": "lorentz", "epsinf": 3.0, "wp": 1.0, "wn": 5.0, "gamma": 0.9},
        }
        y_data = _fit_df(
            lam,
            **{
                "n (sample1)": target.real,
                "k (sample1)": target.imag,
                "n (sample2)": target.real,
                "k (sample2)": target.imag,
            },
        )

        fit, res = nk.fit_to_oscillator(lam, y_data, guess)

        self.assertLess(res.cost, 1e-12)
        np.testing.assert_allclose(
            [fit.model["lorentz_1"][name] for name in ("epsinf", "wp", "wn", "gamma")],
            [1.0, 4.0, 2.2, 0.18],
            rtol=1e-6,
            atol=1e-8,
        )

    def test_direct_mode_accepts_flexible_prefix_columns(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        target = nk.multi_oscillator(lam, oscillator)
        y_data = _fit_df(lam, n_sample1=target.real, k_sample1=target.imag)

        _, res = nk.fit_to_oscillator(
            lam,
            y_data,
            oscillator,
            fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
        )

        self.assertLess(res.cost, 1e-20)

    def test_direct_mode_rejects_missing_and_duplicate_pairs(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}

        with self.assertRaisesRegex(ValueError, "both n and k"):
            nk.fit_to_oscillator(lam, _fit_df(lam, **{"n (sample1)": np.ones_like(lam)}), oscillator)

        duplicate = pd.DataFrame(
            np.column_stack([np.ones_like(lam), np.ones_like(lam), np.ones_like(lam)]),
            columns=["n sample1", "n_sample1", "k sample1"],
            index=lam,
        )
        duplicate.index.name = "wavelength"
        with self.assertRaisesRegex(ValueError, "Duplicate direct-mode n column"):
            nk.fit_to_oscillator(lam, duplicate, oscillator)

    def test_custom_mode_preserves_fit_extra_outputs(self):
        lam = np.linspace(0.5, 0.9, 16)
        oscillator = {"drude_core": {"type": "drude", "epsinf": 2.1, "wp": 4.2, "gamma": 0.15}}
        fixed = {"drude_core": ["epsinf", "wp", "gamma"]}

        def y_eval(lam, nk_vals, scale, shift):
            real_part = scale * nk_vals.real + shift
            imag_part = scale * nk_vals.imag
            return [real_part, imag_part]

        target_nk = nk.multi_oscillator(lam, oscillator)
        target = y_eval(lam, target_nk, scale=1.35, shift=0.08)
        y_data = _fit_df(lam, real_part=target[0], imag_part=target[1])
        fit_extra = {
            "scale": {"init": 0.8, "bounds": (0.5, 2.0)},
            "shift": {"init": -0.1, "bounds": (-0.5, 0.5)},
        }

        _, res = nk.fit_to_oscillator(
            lam,
            y_data,
            oscillator,
            y_eval=y_eval,
            fit_extra_params=fit_extra,
            fixed_params=fixed,
        )

        self.assertIn("scale", res.fit_extra_params)
        self.assertIn("shift", res.fit_extra_params)
        np.testing.assert_allclose(
            [res.fit_extra_params["scale"], res.fit_extra_params["shift"]],
            [1.35, 0.08],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_fixed_parameters_stay_unchanged(self):
        lam = np.linspace(0.6, 1.3, 30)
        target = nk.multi_oscillator(
            lam,
            {"lorentz_1": {"type": "lorentz", "epsinf": 1.0, "wp": 4.1, "wn": 2.6, "gamma": 0.22}},
        )
        guess = {
            "lorentz_1": {"type": "lorentz", "epsinf": 2.5, "wp": 2.5, "wn": 2.1, "gamma": 0.33},
        }
        y_data = _fit_df(lam, n=target.real, k=target.imag)

        fit, _ = nk.fit_to_oscillator(
            lam,
            y_data,
            guess,
            fixed_params={"lorentz_1": ["gamma"]},
        )

        self.assertEqual(fit.model["lorentz_1"]["gamma"], 0.33)

    def test_results_are_deterministic(self):
        lam = np.linspace(0.55, 1.15, 24)
        target = nk.multi_oscillator(
            lam,
            {"lorentz_1": {"type": "lorentz", "epsinf": 1.2, "wp": 3.8, "wn": 2.4, "gamma": 0.25}},
        )
        guess = {
            "lorentz_1": {"type": "lorentz", "epsinf": 2.9, "wp": 1.2, "wn": 5.6, "gamma": 0.7},
        }
        y_data = _fit_df(lam, n=target.real, k=target.imag)

        _, res_a = nk.fit_to_oscillator(lam, y_data, guess)
        _, res_b = nk.fit_to_oscillator(lam, y_data, guess)

        np.testing.assert_allclose(res_a.x, res_b.x, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(res_a.cost, res_b.cost, places=15)

    def test_high_dimensional_fit_extra_params(self):
        lam = np.linspace(0.45, 0.75, 9)
        oscillator = {"drude_core": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        fixed = {"drude_core": ["epsinf", "wp", "gamma"]}

        param_names = [f"p{i}" for i in range(9)]

        def y_eval(lam, nk_vals, **kwargs):
            return [np.array([kwargs[name] for name in param_names], dtype=float)]

        fit_extra = {
            name: {"init": 0.8, "bounds": (0.0, 1.5)}
            for name in param_names
        }
        target = np.linspace(0.1, 0.9, len(param_names))
        y_data = _fit_df(lam, target=target)

        _, res = nk.fit_to_oscillator(
            lam,
            y_data,
            oscillator,
            y_eval=y_eval,
            fit_extra_params=fit_extra,
            fixed_params=fixed,
        )

        np.testing.assert_allclose(res.fit_extra_flat, target, atol=1e-8, rtol=1e-8)

    def test_requires_dataframe_y_data(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}

        with self.assertRaisesRegex(TypeError, "y_data must be a pandas DataFrame"):
            nk.fit_to_oscillator(lam, [np.ones_like(lam), np.zeros_like(lam)], oscillator)

    def test_requires_wavelength_index_name(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = pd.DataFrame({"n": np.ones_like(lam), "k": np.zeros_like(lam)}, index=lam)

        with self.assertRaisesRegex(ValueError, "index name must be 'wavelength'"):
            nk.fit_to_oscillator(lam, y_data, oscillator)

    def test_requires_1d_numpy_wavelength(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam, n=np.ones_like(lam), k=np.zeros_like(lam))

        with self.assertRaisesRegex(TypeError, "wavelength must be a 1D numpy.ndarray"):
            nk.fit_to_oscillator(list(lam), y_data, oscillator)
        with self.assertRaisesRegex(ValueError, "wavelength must be a 1D numpy.ndarray"):
            nk.fit_to_oscillator(lam.reshape(2, 4), y_data, oscillator)

    def test_exact_matching_wavelength_index_skips_interpolation(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        target = nk.multi_oscillator(lam, oscillator)
        y_data = _fit_df(lam, n=target.real, k=target.imag)

        with mock.patch("empylib.nklib._np.interp", side_effect=AssertionError("unexpected interpolation")):
            _, res = nk.fit_to_oscillator(
                lam,
                y_data,
                oscillator,
                fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
            )

        self.assertLess(res.cost, 1e-20)

    def test_exact_unsorted_wavelength_index_skips_sort_and_interpolation(self):
        lam_sorted = np.linspace(0.5, 0.9, 8)
        order = np.array([3, 0, 7, 1, 6, 2, 5, 4])
        lam = lam_sorted[order]
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        target = nk.multi_oscillator(lam, oscillator)
        y_data = _fit_df(lam, n=target.real, k=target.imag)

        with mock.patch("empylib.nklib._np.interp", side_effect=AssertionError("unexpected interpolation")):
            _, res = nk.fit_to_oscillator(
                lam,
                y_data,
                oscillator,
                fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
            )

        self.assertLess(res.cost, 1e-20)

    def test_unsorted_dataframe_index_is_sorted_for_interpolation(self):
        lam_measured = np.linspace(0.5, 0.9, 8)
        lam_fit = np.linspace(0.55, 0.85, 5)
        order = np.array([3, 0, 7, 1, 6, 2, 5, 4])
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}

        def y_eval(lam, nk_vals):
            return [2.0 * lam + 1.0]

        y_data = _fit_df(lam_measured[order], target=2.0 * lam_measured[order] + 1.0)
        _, res = nk.fit_to_oscillator(
            lam_fit,
            y_data,
            oscillator,
            y_eval=y_eval,
            fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
        )

        self.assertLess(res.cost, 1e-20)

    def test_smaller_wavelength_range_interpolates_y_data(self):
        lam_measured = np.linspace(0.5, 1.5, 11)
        lam_fit = np.linspace(0.7, 1.2, 6)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}

        def y_eval(lam, nk_vals):
            return [3.0 * lam - 0.25]

        y_data = _fit_df(lam_measured, target=3.0 * lam_measured - 0.25)
        _, res = nk.fit_to_oscillator(
            lam_fit,
            y_data,
            oscillator,
            y_eval=y_eval,
            fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
        )

        self.assertLess(res.cost, 1e-20)

    def test_wavelength_range_cannot_extrapolate(self):
        lam_measured = np.linspace(0.5, 0.9, 8)
        lam_fit = np.linspace(0.4, 0.8, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam_measured, n=np.ones_like(lam_measured), k=np.zeros_like(lam_measured))

        with self.assertRaisesRegex(ValueError, "extrapolation is not allowed"):
            nk.fit_to_oscillator(lam_fit, y_data, oscillator)

    def test_custom_y_eval_uses_dataframe_column_order(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}

        def y_eval(lam, nk_vals):
            return [lam + 3.0, lam + 1.0]

        y_data = _fit_df(lam, B=lam + 3.0, A=lam + 1.0)
        _, res = nk.fit_to_oscillator(
            lam,
            y_data,
            oscillator,
            y_eval=y_eval,
            fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
        )

        self.assertLess(res.cost, 1e-20)

    def test_interactive_guess_preserves_dict_input_and_updates_model(self):
        lam = np.linspace(0.6, 1.1, 10)
        oscillator = {
            "lorentz_1": {"type": "lorentz", "epsinf": 1.0, "wp": 4.0, "wn": 2.4, "gamma": 0.2},
        }
        target = nk.multi_oscillator(lam, oscillator)
        y_data = _fit_df(lam, n=target.real, k=target.imag)
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            controller = nk.interactive_oscillator_guess(lam, y_data, oscillator)

        self.assertEqual(list(controller.model.keys()), ["lorentz_1"])
        self.assertIsNotNone(controller.widget)
        self.assertEqual(len(controller._sliders), 4)
        self.assertEqual(len(controller.widget.children), 2)
        self.assertIsInstance(controller.widget.children[0], _FakeHBox)
        self.assertEqual(len(controller.widget.children[0].children), 5)
        self.assertEqual(controller.widget.children[0].children[0].value, "lorentz_1:")
        self.assertEqual(
            [slider.description for slider in controller.widget.children[0].children[1:]],
            ["epsinf:", "wp:", "wn:", "gamma:"],
        )
        self.assertEqual(pyplot_mod.subplot_calls, [(1, 2, False, {})])
        self.assertEqual(len(pyplot_mod.created_axes), 2)
        data_ax, nk_ax = pyplot_mod.created_axes
        self.assertIsNotNone(controller.fig)
        self.assertEqual(controller.ax, (data_ax, nk_ax))
        self.assertEqual(controller.mpl_ax, (data_ax, nk_ax))
        self.assertIsInstance(controller.widget.children[-1], _FakeOutput)
        self.assertEqual(controller.output.clear_output_calls, [((), {"wait": True})])
        self.assertEqual(pyplot_mod.display_calls, [controller.fig])
        self.assertEqual(len(data_ax.plot_calls), 2)
        self.assertEqual(data_ax.plot_calls[0][0][2], "-")
        self.assertEqual(data_ax.plot_calls[0][1]["label"], "k")
        self.assertEqual(data_ax.plot_calls[1][0][2], "--k")
        self.assertEqual(data_ax.plot_calls[1][1]["label"], "k (fit)")
        self.assertEqual(data_ax.ylabel, "k")
        self.assertEqual(data_ax.xlabel, "wavelength (um)")
        self.assertEqual(data_ax.yscale, "log")
        self.assertEqual(data_ax.legend_calls, 1)
        self.assertEqual(len(nk_ax.plot_calls), 2)
        self.assertEqual(nk_ax.plot_calls[0][1]["label"], "n")
        self.assertEqual(nk_ax.plot_calls[0][1]["color"], "C0")
        self.assertEqual(nk_ax.plot_calls[1][0][2], "--k")
        self.assertEqual(nk_ax.plot_calls[1][1]["label"], "n (fit)")
        self.assertEqual(nk_ax.ylabel, "n")
        self.assertEqual(nk_ax.xlabel, "wavelength (um)")
        self.assertEqual(nk_ax.yscale, "linear")
        self.assertEqual(nk_ax.legend_calls, 1)
        controller.ax[0].set_ylim(0, 0.1)
        self.assertEqual(controller.ax[0].get_ylim(), (0, 0.1))
        self.assertEqual(data_ax.ylim, (0, 0.1))
        self.assertEqual(len(pyplot_mod.display_calls), 2)
        old_model_ydata = np.asarray(data_ax.lines[1].ydata).copy()
        controller._sliders[1]["slider"].value = 6.25
        self.assertAlmostEqual(controller.get_model()["lorentz_1"]["wp"], 6.25)
        self.assertFalse(np.allclose(old_model_ydata, data_ax.lines[1].ydata))
        self.assertEqual(pyplot_mod.subplot_calls, [(1, 2, False, {})])
        self.assertEqual(data_ax.ylim, (0, 0.1))
        self.assertEqual(controller.fig.canvas.draw_idle_calls, 3)
        self.assertEqual(len(controller.output.clear_output_calls), 3)
        self.assertEqual(len(pyplot_mod.display_calls), 3)
        controller.close()
        self.assertTrue(all(item["slider"].closed for item in controller._sliders))
        self.assertTrue(controller.output.closed)
        self.assertEqual(pyplot_mod.closed_figures, [((controller.fig,), {})])

    def test_interactive_guess_normalizes_list_input_and_respects_fixed_params(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = [
            {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18},
            {"type": "lorentz", "epsinf": 1.2, "wp": 3.0, "wn": 2.1, "gamma": 0.15},
        ]
        target = nk.multi_oscillator(lam, {
            "drude_1": oscillator[0],
            "lorentz_1": oscillator[1],
        })
        y_data = _fit_df(lam, n=target.real, k=target.imag)
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            controller = nk.interactive_oscillator_guess(
                lam,
                y_data,
                oscillator,
                fixed_params={"lorentz_1": ["gamma"]},
            )

        self.assertEqual(list(controller.model.keys()), ["drude_1", "lorentz_1"])
        slider_names = [item["name"] for item in controller._sliders]
        self.assertNotIn("lorentz_1:gamma", slider_names)
        self.assertEqual(len(controller._sliders), 6)
        self.assertEqual(len(controller.widget.children), 3)
        self.assertTrue(all(isinstance(row, _FakeHBox) for row in controller.widget.children[:2]))
        self.assertEqual([len(row.children) for row in controller.widget.children[:2]], [4, 4])
        self.assertEqual([row.children[0].value for row in controller.widget.children[:2]], ["drude_1:", "lorentz_1:"])
        self.assertEqual(
            [[slider.description for slider in row.children[1:]] for row in controller.widget.children[:2]],
            [["epsinf:", "wp:", "gamma:"], ["epsinf:", "wp:", "wn:"]],
        )
        self.assertEqual(pyplot_mod.subplot_calls, [(1, 2, False, {})])

    def test_interactive_guess_direct_mode_plots_multiple_samples_on_split_axes(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        target = nk.multi_oscillator(lam, oscillator)
        y_data = _fit_df(
            lam,
            **{
                "n (sample1)": target.real,
                "k (sample1)": target.imag,
                "n_sample2": target.real + 0.1,
                "k_sample2": target.imag * 1.1,
            },
        )
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            controller = nk.interactive_oscillator_guess(
                lam,
                y_data,
                oscillator,
                fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
            )

        k_ax, n_ax = controller.ax
        self.assertEqual(k_ax.get_yscale(), "log")
        self.assertEqual(n_ax.get_yscale(), "linear")
        self.assertEqual([call[1]["label"] for call in k_ax.plot_calls], ["k (sample1)", "k_sample2", "k (fit)"])
        self.assertEqual([call[1]["label"] for call in n_ax.plot_calls], ["n (sample1)", "n_sample2", "n (fit)"])
        self.assertEqual(n_ax.plot_calls[0][1]["color"], "C0")
        self.assertEqual(n_ax.plot_calls[1][1]["color"], "C1")
        self.assertEqual(k_ax.plot_calls[-1][0][2], "--k")
        self.assertEqual(n_ax.plot_calls[-1][0][2], "--k")
        controller.close()

    def test_interactive_guess_validates_custom_output_shape(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        def y_eval(lam, nk_vals):
            return [nk_vals.real]
        y_data = _fit_df(lam, R=np.ones_like(lam), T=np.ones_like(lam))

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            with self.assertRaisesRegex(ValueError, "y_eval returned 1 outputs"):
                nk.interactive_oscillator_guess(
                    lam,
                    y_data,
                    oscillator,
                    y_eval=y_eval,
                )

    def test_interactive_guess_rejects_legacy_list_y_data(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            with self.assertRaisesRegex(TypeError, "y_data must be a pandas DataFrame"):
                nk.interactive_oscillator_guess(
                    lam,
                    [np.ones_like(lam), np.zeros_like(lam)],
                    oscillator,
                )

    def test_interactive_guess_requires_1d_numpy_wavelength(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam, n=np.ones_like(lam), k=np.zeros_like(lam))
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            with self.assertRaisesRegex(TypeError, "wavelength must be a 1D numpy.ndarray"):
                nk.interactive_oscillator_guess(list(lam), y_data, oscillator)
            with self.assertRaisesRegex(ValueError, "wavelength must be a 1D numpy.ndarray"):
                nk.interactive_oscillator_guess(lam.reshape(2, 4), y_data, oscillator)

    def test_interactive_guess_requires_wavelength_index_name(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = pd.DataFrame({"n": np.ones_like(lam), "k": np.zeros_like(lam)}, index=lam)
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            with self.assertRaisesRegex(ValueError, "index name must be 'wavelength'"):
                nk.interactive_oscillator_guess(lam, y_data, oscillator)

    def test_interactive_guess_default_mode_requires_n_and_k_columns(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam, n=np.ones_like(lam), extinction=np.zeros_like(lam))
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            with self.assertRaisesRegex(ValueError, "must start with n or k"):
                nk.interactive_oscillator_guess(lam, y_data, oscillator)

    def test_interactive_guess_custom_y_eval_uses_dataframe_column_order(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam, B=lam + 3.0, A=lam + 1.0)
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        def y_eval(lam, nk_vals):
            return [lam + 3.0, lam + 1.0]

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            controller = nk.interactive_oscillator_guess(
                lam,
                y_data,
                oscillator,
                y_eval=y_eval,
                fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
            )

        data_ax, _ = controller.ax
        self.assertEqual(data_ax.plot_calls[0][1]["label"], "B data")
        self.assertEqual(data_ax.plot_calls[1][1]["label"], "B model")
        self.assertEqual(data_ax.plot_calls[2][1]["label"], "A data")
        self.assertEqual(data_ax.plot_calls[3][1]["label"], "A model")
        controller.close()

    def test_interactive_guess_smaller_wavelength_range_plots_raw_data_grid(self):
        lam_measured = np.linspace(0.5, 1.5, 11)
        lam_fit = np.linspace(0.7, 1.2, 6)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam_measured, target=3.0 * lam_measured - 0.25)
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        def y_eval(lam, nk_vals):
            return [3.0 * lam - 0.25]

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            controller = nk.interactive_oscillator_guess(
                lam_fit,
                y_data,
                oscillator,
                y_eval=y_eval,
                fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
            )

        data_ax, _ = controller.ax
        np.testing.assert_allclose(data_ax.plot_calls[0][0][0], lam_measured)
        np.testing.assert_allclose(data_ax.plot_calls[0][0][1], 3.0 * lam_measured - 0.25)
        np.testing.assert_allclose(data_ax.plot_calls[1][0][0], lam_fit)
        self.assertEqual(data_ax.xlim, (float(lam_measured.min()), float(lam_measured.max())))
        self.assertEqual(controller.ax[1].xlim, (float(lam_measured.min()), float(lam_measured.max())))
        self.assertEqual(controller.fig.title, "Interactive oscillator guess")
        controller.close()

    def test_interactive_guess_rejects_y_eval_output_index_mismatch(self):
        lam_measured = np.linspace(0.25, 2.5, 300)
        lam_fit = np.linspace(0.5, 2.5, 300)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam_measured, target=np.ones_like(lam_measured))
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        def y_eval(lam, nk_vals):
            return [pd.Series(np.ones_like(lam), index=lam_measured)]

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            with self.assertRaisesRegex(ValueError, "index that does not match wavelength"):
                nk.interactive_oscillator_guess(
                    lam_fit,
                    y_data,
                    oscillator,
                    y_eval=y_eval,
                    fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
                )

    def test_interactive_guess_never_interpolates_y_data(self):
        lam_measured = np.linspace(0.5, 1.5, 11)
        lam_fit = np.linspace(0.7, 1.2, 6)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam_measured, n=np.ones_like(lam_measured), k=np.zeros_like(lam_measured))
        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            with mock.patch("empylib.nklib._np.interp", side_effect=AssertionError("unexpected interpolation")):
                controller = nk.interactive_oscillator_guess(
                    lam_fit,
                    y_data,
                    oscillator,
                    fixed_params={"drude_1": ["epsinf", "wp", "gamma"]},
                )

        controller.close()

    def test_interactive_guess_rejects_old_keyword_names(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}
        y_data = _fit_df(lam, n=np.ones_like(lam), k=np.zeros_like(lam))

        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'x_units'"):
            nk.interactive_oscillator_guess(lam, y_data, oscillator, x_units="um")
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'slider_bounds'"):
            nk.interactive_oscillator_guess(lam, y_data, oscillator, slider_bounds={})

    def test_interactive_guess_raises_clear_dependency_errors(self):
        lam = np.linspace(0.5, 0.9, 8)
        oscillator = {"drude_1": {"type": "drude", "epsinf": 2.0, "wp": 4.5, "gamma": 0.18}}

        real_import = __import__

        def missing_ipywidgets(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ipywidgets":
                raise ImportError("missing ipywidgets")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=missing_ipywidgets):
            with self.assertRaisesRegex(ImportError, "requires ipywidgets"):
                nk.interactive_oscillator_guess(
                    lam,
                    _fit_df(lam, n=np.ones_like(lam), k=np.ones_like(lam)),
                    oscillator,
                )

        widgets_mod, matplotlib_mod, pyplot_mod = _make_fake_widget_modules()

        def missing_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "matplotlib.pyplot":
                raise ImportError("missing matplotlib")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.dict(sys.modules, _fake_interactive_modules(widgets_mod, matplotlib_mod, pyplot_mod)):
            with mock.patch("builtins.__import__", side_effect=missing_matplotlib):
                with self.assertRaisesRegex(ImportError, "requires matplotlib"):
                    nk.interactive_oscillator_guess(
                        lam,
                        _fit_df(lam, n=np.ones_like(lam), k=np.ones_like(lam)),
                        oscillator,
                    )


if __name__ == "__main__":
    unittest.main()
