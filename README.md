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
| 5 | Adaptive policy engine + deterministic safety governor | done |
| 6 | Conventional vs adaptive comparison | done (simulation) |

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

Phases 5-6 (adaptive PLCA vs conventional), about 2-4 minutes:

```
python run_adaptive.py
```

Real traffic from the ROAD dataset (unzip it to `dataset\road\`):

```
python run_road.py
```

## Real traffic: ROAD CAN traces (`run_road.py`, `plca_sim/road.py`)

Replays the attack-free ROAD log `ambient_highway_street_driving_long.log`
(choose another with `--log`) on the 10BASE-T1S zone. Each CAN frame keeps
its real timestamp and becomes a 64-byte Ethernet frame; CAN IDs are spread
over the 8 nodes by load (ROAD's DBC is anonymized, so the sending ECU is
unknown); the two fastest IDs are treated as safety-critical (2 ms
deadline); replaying the trace k times with random offsets models a zone
that aggregates k CAN segments.

* Experiment A: CAN traffic only, 1-8 CAN segments.
* Experiment B: CAN traffic + 30 % bulk Ethernet with large frames
  (diagnostics, software updates), 1-4 CAN segments.

The governor uses the largest frame that actually occurs (64 bytes in A), so
its guarantee stays feasible for many CAN segments.

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

## Phase 5: adaptive policy engine + safety governor

`plca_sim/adaptive.py`

* **Policy engine** (`AdaptivePolicy`): low/medium state -> conventional
  PLCA; high -> nodes with >=3 queued frames may send 2 per TO; congested ->
  nodes with >=2 queued frames may send up to 8 per TO. The state comes from
  the TinyML model (`mode="tinyml"`), from the measured current state
  (`"reactive"`), or adaptation is off (`"off"`).
* **Safety governor** (`SafetyGovernor`): checks every plan before it reaches
  the bus and never trusts the prediction. For known critical flows
  (period P_i, frame time c_i, deadline D) it fixes L_i = ceil(D/P_i)
  critical frames per TO and a budget B for non-critical bus time counted
  since each critical node's last TO. The gap between two TOs of a critical
  node is then at most C = F + sum L_i c_i + B, so every critical frame is
  delivered within C + L_i c_i <= D (proof in the class docstring; checked
  by `tests/test_adaptive.py`). Without critical flows it caps the gap at
  conventional PLCA's own worst case. All nodes see the bus, so each node
  can enforce the budget locally.

## Phase 6 results (`run_adaptive.py`, unseen seeds 200-202)

Mixed criticality: the governor guarantees a 1.9 ms bound for the 2 ms
deadline. Measured worst critical latency is about 1.45 ms at every load,
and **0 % deadline misses** vs. 0.5 / 1.8 / 16 / 42 / 78 % for conventional
PLCA at 20 / 30 / 50 / 70 / 90 % load. Frame loss at 90 % drops from 15 % to
0.2 %. The price: non-critical frames wait longer (average 17.9 -> 32.9 ms
at 90 % load).

Change of adaptive (TinyML) vs conventional PLCA at 90 % load:

| Scenario | Mean latency | p99 latency | Jitter | Throughput |
|---|---|---|---|---|
| uniform | -5 % | -11 % | -9 % | 0 % |
| heterogeneous | -23 % | -42 % | +48 % | 0 % |
| bursty | -6 % | +5 % | -12 % | 0 % |
| dynamic | -3 % | -15 % | -8 % | 0 % |

Honest observations:

* The large, guaranteed gain is for safety-critical traffic, and it comes
  from the governor. Burst adaptation gives moderate gains for the rest.
* TinyML-driven adaptation performs almost the same as reactive adaptation
  (measured state): with ~62 % prediction accuracy (Phase 4) the prediction
  adds little over measuring the current state.
* With unequal ECU loads, bursts cut latency but raise jitter for the
  lightly loaded ECUs.

## Project layout

```
plca_sim/
  config.py      PLCA parameters (nodes, TO timer, beacon, burst mode, queue size)
  traffic.py     Poisson, bursty and periodic (safety-critical) traffic generators
  scenarios.py   Evaluation scenarios 1-4 plus "dynamic"
  simulator.py   Cycle-level PLCA simulator (BEACON -> TO 0..N-1 -> next BEACON)
  policy.py      Policy interface; ConventionalPolicy = standard PLCA
  metrics.py     KPIs from a simulation run
  monitor.py     PLCA monitoring and feature extraction
  tinyml.py      Labelling, decision-tree training, export to C
  adaptive.py    Adaptive policy engine and safety governor
  road.py        ROAD candump reader and CAN -> 10BASE-T1S trace replay
run_baseline.py  Runs all scenarios with conventional PLCA and plots results
run_tinyml.py    Builds the dataset, trains and evaluates the TinyML model
run_adaptive.py  Compares conventional, governor, reactive and TinyML policies
run_road.py      Same comparison on real ROAD CAN traffic
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
