"""Checks for the safety governor and the adaptive policy."""

import unittest

from plca_sim import PLCAConfig, make_traffic, simulate, summarize
from plca_sim.adaptive import AdaptivePolicy, GovernorInfeasible, SafetyGovernor
from plca_sim.scenarios import critical_flows
from plca_sim.traffic import CriticalFlow


class TestGovernor(unittest.TestCase):
    def setUp(self):
        self.cfg = PLCAConfig()

    def test_critical_bound_holds_under_overload(self):
        cfg = self.cfg
        gov = SafetyGovernor(cfg, critical_flows("mixed"))
        bound = max(gov.latency_bound_bt.values())
        self.assertLessEqual(bound, cfg.s_to_bt(2e-3))
        for seed in (0, 1):
            frames = make_traffic("mixed", 0.95, cfg, 0.5, seed)
            for mode in ("off", "reactive"):
                pol = AdaptivePolicy(cfg, SafetyGovernor(cfg, critical_flows("mixed")), mode=mode)
                res = simulate(frames, cfg, cfg.s_to_bt(0.5), pol)
                worst = max(r.end - r.arrival for r in res.records if r.critical)
                self.assertLessEqual(worst, bound)
                self.assertEqual(summarize(res)["critical_deadline_miss_rate"], 0.0)

    def test_gap_never_exceeds_conventional_worst_case(self):
        cfg = self.cfg
        frames = make_traffic("bursty", 0.95, cfg, 0.5, 3)
        pol = AdaptivePolicy(cfg, SafetyGovernor(cfg), mode="reactive", max_burst=16)
        res = simulate(frames, cfg, cfg.s_to_bt(0.5), pol)
        gov = SafetyGovernor(cfg)
        self.assertLessEqual(res.max_to_gap_bt, gov.worst_case_gap_bt)
        conv = simulate(frames, cfg, cfg.s_to_bt(0.5))
        self.assertLessEqual(conv.max_to_gap_bt, gov.worst_case_gap_bt)

    def test_infeasible_deadline_rejected(self):
        flow = CriticalFlow(1, 1e-3, 64, 0.5e-3)
        with self.assertRaises(GovernorInfeasible):
            SafetyGovernor(self.cfg, [flow])

    def test_mode_off_without_governor_is_conventional(self):
        cfg = self.cfg
        frames = make_traffic("uniform", 0.6, cfg, 0.2, 0)
        a = simulate(frames, cfg, cfg.s_to_bt(0.2))
        b = simulate(frames, cfg, cfg.s_to_bt(0.2), AdaptivePolicy(cfg, mode="off"))
        self.assertEqual([r.end for r in a.records], [r.end for r in b.records])


if __name__ == "__main__":
    unittest.main()
