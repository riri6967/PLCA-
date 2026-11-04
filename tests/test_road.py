"""Checks for the ROAD candump reader and trace replay."""

import os
import tempfile
import unittest

import numpy as np

from plca_sim import PLCAConfig, simulate, summarize
from plca_sim.adaptive import AdaptivePolicy, SafetyGovernor
from plca_sim.road import id_stats, make_road_traffic, max_in_window, read_candump, select_critical_ids


def write_log(lines):
    fd, path = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


class TestRoad(unittest.TestCase):
    def test_parse_candump_formats(self):
        path = write_log([
            "(1000000000.000000) can0 0C1#00000000FFFFFFFF",
            "(1000000000.010000) can0 0C1#00000000FFFFFFFF",
            "garbage line",
            "(1000000000.012000) can0 1A2#0102",
            "(1000000000.013000) can0 12345678#R",
        ])
        df = read_candump(path)
        os.remove(path)
        self.assertEqual(list(df.can_id), [0x0C1, 0x0C1, 0x1A2, 0x12345678])
        self.assertEqual(list(df.dlc), [8, 8, 2, 0])
        self.assertAlmostEqual(df.t.iloc[1], 0.01)

    def test_replay_scales_load_and_marks_critical(self):
        lines = []
        for k in range(2000):  # 10 s
            t = k * 0.005
            lines.append(f"({t:.6f}) can0 0A0#1122334455667788")
            if k % 10 == 0:
                lines.append(f"({t + 0.001:.6f}) can0 3F0#11")
        path = write_log(lines)
        trace = read_candump(path)
        os.remove(path)
        self.assertEqual(id_stats(trace).can_id.iloc[0], 0x0A0)
        cfg = PLCAConfig()
        one = make_road_traffic(trace, cfg, 1, 2.0, 0, n_critical=1)
        three = make_road_traffic(trace, cfg, 3, 2.0, 0, n_critical=1)
        self.assertAlmostEqual(three.offered_load / one.offered_load, 3.0, delta=0.1)
        self.assertEqual(len(three.flows), 3)
        self.assertTrue(all(f.critical for f in one.frames if f.size == 64 and f.deadline))
        self.assertEqual(one.max_frame_bytes, 64)

    def test_aperiodic_fastest_id_is_not_critical(self):
        lines = []
        for k in range(4000):
            t = k * 0.01
            lines.append(f"({t:.6f}) can0 0FF#11")
            lines.append(f"({t + 0.002:.6f}) can0 0AA#11")
            lines.append(f"({t + 0.004:.6f}) can0 FFF#00")
            lines.append(f"({t + 0.007:.6f}) can0 FFF#00")
            if k % 10 == 0:
                lines.append(f"({t + 0.007001:.6f}) can0 FFF#00")
        path = write_log(lines)
        trace = read_candump(path)
        os.remove(path)
        self.assertEqual(select_critical_ids(id_stats(trace), 2), [0x0AA, 0x0FF])
        self.assertEqual(select_critical_ids(id_stats(trace), 1, skip=1), [0x0FF])

    def test_max_in_window(self):
        self.assertEqual(max_in_window([0.0, 0.5, 0.9, 1.0, 5.0], 1.0), 3)
        self.assertEqual(max_in_window([], 1.0), 0)

    def test_guarantee_holds_with_jitter_and_overload(self):
        """Jittery flows, heavy load: no critical frame lost or late."""
        rng = np.random.default_rng(1)
        lines = []
        for k in range(1500):
            t = k * 0.01
            lines.append(f"({t + rng.uniform(0, 0.003):.6f}) can0 010#1122334455667788")
            lines.append(f"({t + rng.uniform(0, 0.003):.6f}) can0 020#1122334455667788")
            for j in range(20):
                lines.append(f"({t + rng.uniform(0, 0.01):.6f}) can0 {0x100 + j:03X}#1122334455667788")
        lines.sort(key=lambda s: float(s[1:s.index(")")]))
        path = write_log(lines)
        trace = read_candump(path)
        os.remove(path)
        cfg = PLCAConfig()
        rt = make_road_traffic(trace, cfg, 2, 3.0, 0, 2, 2e-3, bulk_load=0.7)
        gov = SafetyGovernor(cfg, rt.flows, max_frame_bytes=rt.max_frame_bytes)
        res = simulate(rt.frames, cfg, cfg.s_to_bt(3.0), AdaptivePolicy(cfg, gov, mode="off"))
        worst = max(r.end - r.arrival for r in res.records if r.critical)
        self.assertLessEqual(worst, max(gov.latency_bound_bt.values()))
        self.assertEqual(summarize(res)["critical_deadline_miss_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
