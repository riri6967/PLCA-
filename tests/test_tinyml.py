"""Checks for monitoring, labelling and the TinyML export."""

import unittest

import numpy as np

from plca_sim import PLCAConfig, make_traffic, simulate
from plca_sim.monitor import FEATURES, MonitoringPolicy
from plca_sim.tinyml import collect_samples, export_c, state_of, train_tree


class TestTinyML(unittest.TestCase):
    def setUp(self):
        self.cfg = PLCAConfig()

    def test_state_thresholds(self):
        s = state_of(np.array([0.1, 0.4, 0.7, 0.9, 0.1]), np.array([0, 0, 0, 0, 50]))
        self.assertEqual(list(s), [0, 1, 2, 3, 3])

    def test_monitor_does_not_change_bus_behaviour(self):
        frames = make_traffic("uniform", 0.5, self.cfg, 0.05, 0)
        a = simulate(frames, self.cfg, self.cfg.s_to_bt(0.05))
        b = simulate(frames, self.cfg, self.cfg.s_to_bt(0.05), MonitoringPolicy(self.cfg))
        self.assertEqual([r.end for r in a.records], [r.end for r in b.records])

    def test_monitor_utilization_matches_load(self):
        pol = MonitoringPolicy(self.cfg)
        frames = make_traffic("uniform", 0.5, self.cfg, 0.5, 1)
        simulate(frames, self.cfg, self.cfg.s_to_bt(0.5), pol)
        util = np.mean([s["util"] for s in pol.monitor.samples])
        self.assertAlmostEqual(util, 0.5, delta=0.08)
        self.assertEqual(set(FEATURES) - set(pol.monitor.samples[0]), set())

    def test_train_and_export(self):
        df = collect_samples("dynamic", 0.9, 0, self.cfg, 0.3)
        self.assertGreater(len(df), 20)
        model = train_tree(df, max_depth=3)
        code = export_c(model)
        self.assertIn("plca_predict_state", code)
        self.assertEqual(code.count("{"), code.count("}"))


if __name__ == "__main__":
    unittest.main()
