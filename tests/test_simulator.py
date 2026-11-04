"""Sanity checks for the PLCA simulator. Run with:  python -m unittest"""

import unittest

from plca_sim import Frame, PLCAConfig, simulate, summarize


class TestPLCA(unittest.TestCase):
    def setUp(self):
        self.cfg = PLCAConfig(node_count=4)

    def test_empty_bus_cycle_length(self):
        res = simulate([], self.cfg, 10_000)
        self.assertEqual(res.cycle_lengths[0], self.cfg.empty_cycle_bt)
        self.assertEqual(res.tos_used, 0)

    def test_single_frame_latency(self):
        # Frame for node 2 arrives at t=0: it waits for the BEACON and the
        # empty TOs of nodes 0 and 1, then is transmitted.
        cfg = self.cfg
        res = simulate([Frame(2, 0.0, 64)], cfg, 10_000)
        rec = res.records[0]
        self.assertEqual(rec.start, cfg.beacon_bt + 2 * cfg.to_timer_bt)
        self.assertEqual(rec.end - rec.start, cfg.wire_bits(64))

    def test_round_robin_one_frame_per_to(self):
        frames = [Frame(1, 0.0, 64) for _ in range(3)]
        res = simulate(frames, self.cfg, 100_000)
        starts = [r.start for r in res.records]
        cycle = self.cfg.empty_cycle_bt + self.cfg.wire_bits(64) - self.cfg.to_timer_bt
        self.assertAlmostEqual(starts[1] - starts[0], cycle)
        self.assertAlmostEqual(starts[2] - starts[1], cycle)

    def test_burst_mode(self):
        cfg = PLCAConfig(node_count=4, max_bc=2)
        frames = [Frame(1, 0.0, 64) for _ in range(3)]
        res = simulate(frames, cfg, 100_000)
        self.assertEqual(len(res.cycle_lengths) > 0, True)
        self.assertEqual(res.records[1].start, res.records[0].end)
        self.assertEqual(res.records[2].start, res.records[1].end)

    def test_critical_served_first(self):
        frames = [Frame(1, 0.0, 1518), Frame(1, 0.0, 64, critical=True, deadline=1e6)]
        res = simulate(frames, self.cfg, 100_000)
        self.assertTrue(res.records[0].critical)

    def test_no_loss_at_low_load(self):
        frames = [Frame(n, float(t), 128) for n in range(4) for t in range(0, 200_000, 5_000)]
        s = summarize(simulate(frames, self.cfg, 300_000))
        self.assertEqual(s["frames_dropped"], 0)
        self.assertEqual(s["frames_delivered"], len(frames))


if __name__ == "__main__":
    unittest.main()
