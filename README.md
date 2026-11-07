# Self-Adaptive PLCA for 10BASE-T1S Zonal Automotive Networks

Research project: show the limits of conventional PLCA (Physical Layer
Collision Avoidance, IEEE 802.3cg / 10BASE-T1S) under dynamic traffic, then
build a self-adaptive PLCA framework (monitoring + TinyML prediction +
adaptive policy + deterministic safety governor) and compare the two.

## Status

Done in simulation; nothing here ran on real 10BASE-T1S hardware.
[Paper draft](https://claude.ai/artifact/Gp8QCyRSX1jDppPn7tdxH8).

| Work Plan phase | Status |
|---|---|
| 0 Literature | done; a few sources were read as records or abstracts only (see the paper) |
| 1 Setup | done: OMNeT++ 6.4.0 and INET 4.7.0 in WSL Ubuntu 24.04 |
| 2 Baseline | done: S1-S4, 4 to 16 followers, 10 seeds |
| 3 Dataset and TinyML | done: 467,844 windows, int8 MLP exported to C |
| 4 Adaptive controller | done: `sim/AdaptivePlcaController.cc`, governor at cycle boundaries |
| 5 Evaluation | done: B1-B4, P, ablation, sensitivity, 1,247 runs |
| 6 Hardware | not done, optional in the plan |
| 7 Paper | draft done; poster fixes still open |

## Main results (OMNeT++, S4 mixed, 90 % load, 20 ms deadline, 10 seeds)

| Design | Safety misses | Worst safety (ms) | p99 (ms) | Mean (ms) |
|---|---|---|---|---|
| B1 conventional | 20.0 % | 221.5 | 69.6 | 8.93 |
| B2 best static | 0 | 14.7 | 17.8 | 3.11 |
| B3 fixed priority | 0 | 8.0 | 19.1 | 2.53 |
| B4 reactive EWMA + governor | 0 | 15.8 | 16.8 | 2.72 |
| P TinyML + governor | 0 | 15.1 | 15.9 | 2.65 |
| A1 P without governor | 0.073 % | 35.3 | 17.3 | 2.83 |

* The governor is the useful part: 0 misses in all 480 governed comparison runs,
  including random predictions, against 35.3 ms without it.
* Prediction adds no detectable gain: P and B4 never differ significantly.
* Adaptation helps little: p99 -16.6 % against B3 at 90 % load in S4, mean +4.6 %,
  and no p99 gain on bursty traffic. PLCA wastes little idle time, so burst
  changes move delay between nodes rather than add capacity.

Everything is in `results/omnet/`; tables and figures in `results/omnet/figures/`.

## Datasets and licences

| Data | Where | Note |
|---|---|---|
| ROAD | `dataset/road.zip` (Git LFS) | Verma et al., PLOS ONE 2024 |
| CarDS | `dataset/cards/`, not in Git | CC BY-SA 4.0; Hellemans et al., IEEE ACSAC 2025, doi:10.1109/ACSAC67867.2025.00069 |
| TOW-IDS | not used | needs an IEEE DataPort subscription |

`results/cards/` is derived from CarDS and is shared under CC BY-SA 4.0.
`results/omnet/tinyml_dataset.csv.gz` is the simulated dataset the plan releases.
Critical flows, deadlines and the 30 % bulk share are assumptions: no dataset gives them.

## Reproduce the OMNeT++ results

Run in WSL Ubuntu. `setup_omnet.sh` builds OMNeT++ and INET, 30-60 minutes and
about 6 GB. Runs take about 45 s each on 12 cores with 10 workers.

```
bash sim/setup_omnet.sh
cd sim
python3 run_batch.py S4_Mixed     # one config; finished runs are skipped
python3 summarize.py results/summary.csv --out results/summary_ci.csv
```

Then on Windows, from the repository root:

```
python -m ml.run_omnet_tinyml
python -m analysis.plot_omnet
python -m traffic.run_cards_profile
python -m traffic.run_cards
```

## Cycle-level simulator: run it on Windows

Open **Command Prompt** in the project folder, then run:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m unittest
python -m analysis.run_baseline
```

Results (CSV + PNG plots) are written to the `results\` folder.
Use `python -m analysis.run_baseline --quick` for a fast check.

Phase 4 (TinyML), about 1 minute:

```
python -m ml.run_tinyml
```

Phases 5-6 (adaptive PLCA vs conventional), about 2-4 minutes:

```
python -m analysis.run_adaptive
```

Real traffic from the ROAD dataset (unzip it to `dataset\road\`):

```
python -m traffic.run_road --seeds 5
python -m ml.run_road_tinyml
python -m analysis.run_tradeoff
python -m ml.run_reference_models
```

## Real traffic: ROAD CAN traces (`traffic/run_road.py`, `plca_sim/road.py`)

Replays all 12 attack-free ROAD logs (choose others with `--log`) on the
10BASE-T1S zone. Each CAN frame keeps its real timestamp and becomes a
64-byte Ethernet frame. ROAD's DBC is anonymized, so the sending ECU is
unknown: critical flows are spread over nodes first and other IDs balanced
by load. Critical IDs (the fastest periodic IDs) and the 2 ms deadline are
assumptions: ROAD does not reveal them, and its ID obfuscation does not
preserve bus priority. Replaying a trace k times with random offsets models a zone
that aggregates k CAN segments.

* Experiment A: CAN traffic only, 1-8 CAN segments.
* Experiment B: CAN traffic + 30 % bulk Ethernet with large frames
  (diagnostics, software updates), 1-5 CAN segments.

Governor: 0 misses in 1,620 runs, worst 1.80 ms against a 1.9 ms bound, and
it refuses the 98 % load it cannot guarantee. Conventional PLCA: misses in
271 of 600 runs at 2 ms, all in experiment B, but none at 5 or 10 ms. The
cost at 2 ms: p99 of all frames doubles at 84 % load. Burst adaptation does
not help on this traffic.

## Cycle-level simulator: TinyML network-state prediction

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
  most 4 comparisons and 208 bytes with gcc -Os on x86-64.

Results on unseen runs: 62 % accuracy / 0.58 macro-F1 vs. 61 % / 0.55 for a
"next = now" persistence baseline. 88 % of predictions are correct or one
class off. Upper bounds: a much larger gradient-boosting model reaches only
63 %, and an oracle that knows the true scenario and load 67 %, so the
remaining errors come mostly from the randomness of the traffic itself.
Predictions alone are therefore not trusted blindly: the Phase 5 safety
governor must enforce the deterministic guarantees whatever the model says.

## Cycle-level simulator: adaptive policy engine + metered governor

`plca_sim/adaptive.py`

* **Policy engine** (`AdaptivePolicy`): low/medium state -> conventional
  PLCA; high -> nodes with >=3 queued frames may send 2 per TO; congested ->
  nodes with >=2 queued frames may send up to 8 per TO. The state comes from
  the TinyML model (`mode="tinyml"`), from the measured current state
  (`"reactive"`), or adaptation is off (`"off"`).
* **Safety governor** (`SafetyGovernor`): checks every plan before it reaches
  the bus and never trusts the prediction. For known critical flows
  (period P_i, frame time c_i, deadline D) it fixes L_i = ceil(D/P_i), or
  the declared worst count for flows with jitter, critical frames per TO and a budget B for non-critical bus time counted
  since each critical node's last TO. The gap between two TOs of a critical
  node is then at most C = F + sum L_i c_i + B, so every critical frame is
  delivered within C + L_i c_i <= D (proof in the class docstring; checked
  by `tests/test_adaptive.py`). Without critical flows it caps the gap at
  conventional PLCA's own worst case. All nodes see the bus, so each node
  can enforce the budget locally.

## Cycle-level simulator results (`analysis/run_adaptive.py --seeds 5`, seeds 200-204)

Mixed criticality: the governor guarantees a 1.9 ms bound for the 2 ms
deadline. Measured worst critical latency is at most 1.50 ms at every load,
and **0 % deadline misses** vs. 0.4 / 2.4 / 19 / 39 / 78 % for conventional
PLCA at 20 / 30 / 50 / 70 / 90 % load. Frame loss at 90 % drops from 11.5 % to
0.1 %. The price: non-critical frames wait longer (average 15.3 -> 27.4 ms
at 90 % load). Misses include dropped and overdue critical frames.

Change of adaptive (TinyML) vs conventional PLCA at 90 % load:

| Scenario | Mean latency | p99 latency | Jitter | Throughput |
|---|---|---|---|---|
| uniform | -5 % | -11 % | -9 % | 0 % |
| heterogeneous | -23 % | -42 % | +48 % | 0 % |
| bursty | -6 % | +5 % | -12 % | 0 % |
| dynamic | -3 % | -15 % | -8 % | 0 % |

Honest observations:

* The large, guaranteed gain is for safety-critical traffic, and it comes
  from the governor. Burst adaptation gives moderate gains on synthetic
  traffic, and none on real ROAD traffic.
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
  plan.py        Standards-only governor in Python
sim/             OMNeT++ model
  omnetpp.ini    All scenarios, baselines, ablation and sensitivity configs
  AdaptivePlcaController.cc  Monitoring, TinyML, policy and governor (C++)
  AdaptivePlca.*, ZonalSegment.ned  PLCA with a settable burst count; the network
  run_batch.py, collect.py, summarize.py  Parallel runs and result reduction
traffic/         run_road.py, run_cards_profile.py, run_cards.py
ml/              run_tinyml.py, run_models.py, run_road_tinyml.py,
                 run_reference_models.py, run_omnet_tinyml.py
analysis/        run_baseline.py, run_adaptive.py, run_tradeoff.py,
                 run_plan_eval.py, plot_omnet.py
paper/           Phase 0 source notes
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
