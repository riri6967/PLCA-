"""Checks for the Work Plan's standard-parameters governor."""

import math
import unittest

from plca_sim import PLCAConfig, make_traffic, simulate, summarize
from plca_sim.metrics import jain_index
from plca_sim.plan import PlanGovernor, PlanPolicy, PlcaSettings, safest_settings
from plca_sim.traffic import CriticalFlow, Frame


def setup(deadline_s, frames=(64, 64, 64, 1518, 1518, 1518, 1518, 1518), period_s=10e-3):
    cfg = PLCAConfig()
    flows = [CriticalFlow(1, period_s, 64, deadline_s), CriticalFlow(2, period_s, 64, deadline_s)]
    return cfg, PlanGovernor(cfg, flows, frames)


class TestPlanGovernor(unittest.TestCase):
    def test_worst_cycle_matches_formula(self):
        cfg, gov = setup(20e-3)
        s = PlcaSettings((0, 1, 0, 0, 0, 0, 0, 2), tuple(range(8)))
        f64, f1518 = cfg.wire_bits(64), cfg.wire_bits(1518)
        want = cfg.beacon_bt + 3 * f64 + f64 + 5 * f1518 + 2 * f1518
        self.assertEqual(gov.worst_cycle_bt(s), want)

    def test_large_frames_make_2_ms_impossible(self):
        _, gov = setup(2e-3)
        self.assertIsNone(safest_settings(gov))

    def test_small_frames_make_2_ms_possible(self):
        _, gov = setup(2e-3, frames=(64, 64, 64, 128, 128, 128, 128, 128))
        self.assertIsNotNone(safest_settings(gov))

    def test_queue_growth_is_unsafe_at_any_deadline(self):
        cfg, gov = setup(1.0, period_s=1e-3)
        base = PlcaSettings.default(8)
        self.assertTrue(math.isinf(gov.latency_bound_bt(base)[1]))
        self.assertFalse(gov.accepts(base))

    def test_wrong_predictions_cannot_break_the_bound(self):
        cfg, gov = setup(15e-3)
        base = safest_settings(gov)
        self.assertIsNotNone(base)
        frames = [f for f in make_traffic("uniform", 0.6, cfg, 0.3, 1) if f.node > 2]
        for k in range(30):
            at = k * 1e-2 * cfg.bit_rate
            frames += [Frame(1, at, 64, True, 15e-3 * cfg.bit_rate), Frame(2, at, 64, True, 15e-3 * cfg.bit_rate)]
        pol = PlanPolicy(cfg, base, gov, predictor=lambda feat: 3, hysteresis=1)
        res = simulate(frames, cfg, cfg.s_to_bt(0.3), pol)
        self.assertEqual(summarize(res)["critical_deadline_miss_rate"], 0.0)
        self.assertGreater(pol.overrides, 0)

    def test_no_governor_applies_the_wrong_plan(self):
        cfg = PLCAConfig()
        pol = PlanPolicy(cfg, PlcaSettings.default(8), None, predictor=lambda feat: 3, hysteresis=1)
        simulate(make_traffic("bursty", 0.8, cfg, 0.1, 1), cfg, cfg.s_to_bt(0.1), pol)
        self.assertGreater(pol.changes, 0)
        self.assertEqual(pol.overrides, 0)

    def test_node_order_changes_who_goes_first(self):
        cfg = PLCAConfig()
        frames = [Frame(n, 0.0, 64) for n in range(1, 8)]

        pol = PlanPolicy(cfg, PlcaSettings((0,) * 8, (0, 7, 6, 5, 4, 3, 2, 1)), None, predictor=lambda f: 0)
        res = simulate(frames, cfg, 1e4, pol)
        self.assertEqual(res.records[0].node, 7)


class TestFairness(unittest.TestCase):
    def test_jain(self):
        self.assertAlmostEqual(jain_index([1, 1, 1, 1]), 1.0)
        self.assertAlmostEqual(jain_index([1, 0, 0, 0]), 0.25)

    def test_warmup_skips_early_frames(self):
        cfg = PLCAConfig()
        frames = [Frame(0, 0.0, 1518), Frame(0, 0.0, 1518), Frame(0, 50000.0, 64)]
        res = simulate(frames, cfg, 1e5)
        self.assertGreater(summarize(res)["latency_max_us"], summarize(res, 40000.0)["latency_max_us"])


if __name__ == "__main__":
    unittest.main()
