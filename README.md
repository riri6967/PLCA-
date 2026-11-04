# Self-Adaptive PLCA for 10BASE-T1S Zonal Automotive Networks

Research project: show the limits of conventional PLCA (Physical Layer
Collision Avoidance, IEEE 802.3cg / 10BASE-T1S) under dynamic traffic, then
build a self-adaptive PLCA framework (monitoring + TinyML prediction +
adaptive policy + deterministic safety governor) and compare the two.

## Status

| Phase | Step | Status |
|---|---|---|
| 2 | 10BASE-T1S PLCA network model (1 coordinator + N nodes) | done |
| 2 | Traffic scenarios: low, increasing, heterogeneous, bursty, mixed-criticality | done |
| 2 | Monitoring of latency, jitter, throughput, bus/TO utilization, queues, loss | done |
| 3 | Conventional PLCA baseline + plots | done |
| 4 | TinyML network-state prediction (monitor, features, decision tree, C export) | done |
| 5 | Adaptive policy engine + safety governor | to do |
| 6 | Conventional vs adaptive comparison | to do |

## Run it on Windows

Open **Command Prompt** in the project folder, then run:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m unittest
python run_baseline.py
```

Results (CSV + PNG plots) are written to the `results\` folder.
Use `python run_baseline.py --quick` for a fast check.

Phase 4 (TinyML), about 1 minute:

```
python run_tinyml.py
```

## Phase 4: TinyML network-state prediction

* `plca_sim/monitor.py` records 15 cheap features every 5 ms (bus and TO
  utilization, arrival rate, queue lengths, critical frames waiting, cycle
  length, trends and smoothed values).
* Each window is labelled with the state of the **next 20 ms**:
  low (<30 % bus use), medium (<60 %), high (<85 %), congested
  (>=85 % or >=10 frames waiting).
* A decision tree (depth 4, 31 nodes) is trained on seeds 0-3 and tested on
  unseen seeds 100-101 in all scenarios, including `dynamic` (load and
  burstiness change every 20-100 ms).
* The tree is exported to `results/plca_state_model.h`, a C function with at
  most 4 comparisons and ~240 bytes of compiled code.

Results on unseen runs: 62 % accuracy / 0.58 macro-F1 vs. 61 % / 0.55 for a
"next = now" persistence baseline. 88 % of predictions are correct or one
class off. Upper bounds: a much larger gradient-boosting model reaches only
63 %, and an oracle that knows the true scenario and load 67 %, so the
remaining errors come mostly from the randomness of the traffic itself.
Predictions alone are therefore not trusted blindly: the Phase 5 safety
governor must enforce the deterministic guarantees whatever the model says.

## Project layout

```
plca_sim/
  config.py      PLCA parameters (nodes, TO timer, beacon, burst mode, queue size)
  traffic.py     Poisson, bursty and periodic (safety-critical) traffic generators
  scenarios.py   Evaluation scenarios 1-4
  simulator.py   Cycle-level PLCA simulator (BEACON -> TO 0..N-1 -> next BEACON)
  policy.py      Policy interface; ConventionalPolicy = standard PLCA
  metrics.py     KPIs from a simulation run
  monitor.py     PLCA monitoring and feature extraction
  tinyml.py      Labelling, decision-tree training, export to C
run_baseline.py  Runs all scenarios with conventional PLCA and plots results
run_tinyml.py    Builds the dataset, trains and evaluates the TinyML model
tests/           Sanity checks
```

## Model assumptions

* 10 Mb/s, 1 bit time (BT) = 100 ns. BEACON = 20 BT, empty TO = 32 BT.
* A frame occupies `(size + 8 preamble bytes) * 8 + 96 IPG` BT on the bus.
* A node with data transmits at the start of its TO; with burst mode
  (`max_bc`) or an adaptive policy it may send several frames per TO.
* Critical frames are served before non-critical frames inside a node.
* Latency = time from frame arrival in the queue to end of its transmission.

The adaptive framework plugs in by subclassing `plca_sim.policy.Policy` and
returning the number of frames each node may send in the next cycle.
