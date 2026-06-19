import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_CORR_DIR = Path(__file__).resolve().parent.parent / "correlations"
if str(_CORR_DIR) not in sys.path:
    sys.path.insert(0, str(_CORR_DIR))
import correlation_comparison as cc  # noqa: E402


class TestCorrelationComparison(unittest.TestCase):
    def test_obtener_tipo_correlacion_nb_format(self):
        fp = "/any/corr_length_real_inverse_nb_1_c_s.txt"
        self.assertEqual(cc.obtener_tipo_correlacion(fp), "c_s")

    def test_obtener_tipo_correlacion_unknown(self):
        fp = "/any/corr_length_something_else.txt"
        self.assertEqual(cc.obtener_tipo_correlacion(fp), "Desconocido")

    def test_scenario_tags_parsing(self):
        scenario = {"ALLEE_TYPE": "STRONG", "mu": "1", "USE_ADAPTIVE_CONTROL": "Y"}
        tags = cc.scenario_tags("strong_mu1_uNo_bajo_umbral", scenario)
        self.assertEqual(tags["allee"], "STRONG")
        self.assertEqual(tags["mu"], "1")
        # el nombre debe dominar sobre USE_ADAPTIVE_CONTROL
        self.assertEqual(tags["u"], "uNo")
        self.assertEqual(tags["umbral"], "bajo")

    def test_scenario_tags_from_steady_states(self):
        scenario = {
            "name": "strong_mu1_uSi_bajo_umbral_c0_s1_i0",
            "steady_states": [
                {
                    "allee_type": "STRONG",
                    "mu": 1.0,
                    "use_adaptive_control": True,
                }
            ],
        }
        tags = cc.scenario_tags(scenario["name"], scenario)
        self.assertEqual(tags["allee"], "STRONG")
        self.assertEqual(tags["mu"], "1")
        self.assertEqual(tags["u"], "uSi")
        self.assertEqual(tags["umbral"], "bajo")

    def test_scenario_tags_infer_from_name_only(self):
        tags = cc.scenario_tags("strong_mu0_uNo_bajo_umbral_c0_s1_i0", {})
        self.assertEqual(tags["allee"], "STRONG")
        self.assertEqual(tags["mu"], "0")
        self.assertEqual(tags["u"], "uNo")

    def test_resolve_T_from_common_params(self):
        class Args:
            t_max = None

        t = cc.resolve_T(Args(), {"T": "1"})
        self.assertEqual(t, 1.0)

    def test_english_variational_and_adaptive_labels(self):
        self.assertIn("Hill", cc._u_adaptive_label("uSi"))
        self.assertIn("no adaptive control", cc._u_adaptive_label("uNo"))
        self.assertIn(r"\mu = 1", cc._mu_variational_label("1"))
        full = cc._series_label("0", "uSi")
        self.assertIn(r"\mu = 0", full)
        self.assertIn("Hill", full)

    def test_figure_suptitle_english(self):
        title = cc._figure_suptitle("s_s")
        self.assertIn("Autocorrelation", title)
        self.assertIn("variational", title)
        self.assertIn("adaptive control", title)
        cross = cc._figure_suptitle("c_s")
        self.assertIn("Cross-correlation", cross)

    def test_scaled_out_path(self):
        p = Path("corr_grid_c_c.png")
        self.assertEqual(cc.scaled_out_path(p, "linear").name, "corr_grid_c_c.png")
        self.assertEqual(cc.scaled_out_path(p, "loglog").name, "corr_grid_c_c_loglog.png")
        self.assertEqual(cc.scaled_out_path(p, "semilogx").name, "corr_grid_c_c_semilogx.png")

    def test_format_power_law_label(self):
        fit = {"m": 0.5, "alpha": 0.0, "r2": 0.95}
        label = cc._format_power_law_label(r"$\mu=0$", fit)
        self.assertIn(r"\mu=0", label)
        self.assertIn("t^{0.5}", label)
        self.assertIn("R^2", label)
        self.assertNotIn("n=", label)

    def test_load_corr_series_clips_tmax(self):
        arr = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "corr.txt"
            np.savetxt(p, arr, delimiter="\t")
            t, y = cc.load_corr_series(p, t_max=1.0)
            self.assertTrue(np.all(t <= 1.0))
            self.assertEqual(len(t), 2)

    def test_load_corr_series_sorted_unique(self):
        # t duplicado y desordenado
        arr = np.array([
            [0.2, 2.0],
            [0.1, 1.0],
            [0.2, 2.1],  # duplicado, debería quedarse el primero según idx de unique tras sort
            [0.3, 3.0],
        ])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "corr.txt"
            np.savetxt(p, arr, delimiter="\t")
            t, y = cc.load_corr_series(p)
            self.assertTrue(np.all(np.diff(t) > 0))
            self.assertEqual(len(t), 3)

    def test_resample_to_grid(self):
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])
        grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        out = cc.resample_to_grid(t, y, grid)
        self.assertAlmostEqual(out[1], 0.5, places=7)
        self.assertAlmostEqual(out[3], 2.5, places=7)

    def test_fit_power_law_fixed_exponent(self):
        # y = exp(a) * t^0.5
        a = 1.23
        t = np.linspace(0.05, 1.9, 200)
        y = np.exp(a) * np.sqrt(t)
        fit = cc.fit_power_law(t, y, exponent=0.5, tmin=0.05, tmax=1.9)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["m"], 0.5, places=6)
        self.assertAlmostEqual(fit["alpha"], a, places=3)
        self.assertTrue(fit["r2"] > 0.999)

    def test_fit_power_law_free_exponent(self):
        a = -0.4
        m = 0.42
        t = np.linspace(0.05, 1.9, 300)
        y = np.exp(a) * (t ** m)
        fit = cc.fit_power_law(t, y, exponent=None, tmin=0.05, tmax=1.9)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["m"], m, places=3)
        self.assertAlmostEqual(fit["alpha"], a, places=2)


if __name__ == "__main__":
    unittest.main()


