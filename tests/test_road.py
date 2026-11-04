"""Checks for the ROAD candump reader and trace replay."""

import os
import tempfile
import unittest

from plca_sim import PLCAConfig
from plca_sim.road import id_stats, make_road_traffic, read_candump


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


if __name__ == "__main__":
    unittest.main()
