# Self-Adaptive PLCA for 10BASE-T1S Zonal Automotive Networks: Project Guide

This guide explains the whole project from the beginning: the goal, the
background, what has been built, how to run every step on Windows, the
results so far, and what is left to do.

---

## 1. The goal in one paragraph

Modern cars are moving to a **zonal** architecture, where many small ECUs
(sensors, lights, motors) in one zone share **one cheap Ethernet cable**
called **10BASE-T1S** (10 Mb/s, IEEE 802.3cg). To share the cable without
collisions, 10BASE-T1S uses **PLCA** (Physical Layer Collision Avoidance), a
fixed round-robin: each node gets one turn per cycle. That is fair and
predictable, but it **ignores how much traffic each node has and how urgent
it is**. This project:

1. shows the performance limits of conventional PLCA under realistic,
   changing traffic (simulation + real car data),
2. proposes a **self-adaptive PLCA framework**: monitoring, feature
   extraction, a TinyML predictor, an adaptive policy engine and a
   **deterministic safety governor** that keeps hard guarantees,
3. compares conventional and adaptive PLCA.

PLCA is **not** PLC programming. No PLC software (TIA Portal, CODESYS) is
needed. Everything is Python.

---

## 2. Background you need for the paper

### 2.1 From distributed to zonal E/E architecture
* Old cars: one ECU per function, each wired separately. Heavy, expensive.
* Zonal cars: a central computer plus zone controllers (front, body, rear).
  Inside each zone many small devices must communicate cheaply.
* 10BASE-T1S: one twisted pair, many nodes ("multidrop"), half duplex,
  10 Mb/s, fewer connectors and PHYs, works with normal Ethernet tools.

### 2.2 How PLCA works
* Node 0 (the **coordinator**) sends a **BEACON** to start a cycle.
* Then each node gets a **Transmit Opportunity (TO)** in node-ID order:
  Node 0, Node 1, ..., Node N-1.
* A node with data transmits in its TO. A node without data stays silent and
  its TO expires after 32 bit times (3.2 µs).
* Then a new BEACON starts the next cycle.
* Result: no collisions, and every node is guaranteed a turn.
* Optional **burst mode** lets a node send more than one frame per TO.

### 2.3 The problem (research question)
* Static round-robin: busy nodes wait, idle TOs are wasted.
* Bursty traffic: a burst of 8 frames needs 8 cycles to clear.
* Mixed criticality: a brake message gets no more priority than a radio
  message. When other nodes send large frames, one cycle can last ~6 ms, so
  a safety message sent every 1 ms piles up.

**Research question:** *How can the system understand current and future
network conditions and adapt its communication policy while maintaining
deterministic and bounded guarantees?*

### 2.4 The proposed framework

```
Automotive application traffic
        |
        v
PLCA monitoring -> feature extraction -> TinyML prediction
                                              |
                                              v
                        adaptive policy engine -> deterministic safety governor
                                                              |
                                                              v
                                                  10BASE-T1S PLCA network
                                                  (feedback to monitoring)
```

---

## 3. What is in the repository

```
PLCA-/
  PROJECT_GUIDE.md   this guide
  README.md          short technical summary
  requirements.txt   Python libraries
  plca_sim/          the simulator and framework (Python package)
    config.py        PLCA parameters (nodes, TO timer, beacon, burst, queue)
    traffic.py       traffic generators: Poisson, bursty, periodic critical
    scenarios.py     evaluation scenarios (uniform, heterogeneous, bursty,
                     mixed, dynamic)
    simulator.py     cycle-level PLCA simulator
    policy.py        policy interface; ConventionalPolicy = standard PLCA
    metrics.py       KPIs: latency, jitter, throughput, utilization, loss...
    monitor.py       PLCA monitoring + feature extraction
    tinyml.py        labelling, decision-tree training, export to C
    adaptive.py      adaptive policy engine + safety governor
    road.py          ROAD dataset reader and CAN -> 10BASE-T1S replay
  run_baseline.py    Phase 3: conventional PLCA baseline
  run_tinyml.py      Phase 4: TinyML predictor
  run_adaptive.py    Phases 5-6: conventional vs adaptive
  run_road.py        real traffic from the ROAD dataset
  tests/             automatic checks (python -m unittest)
  results/           outputs (created when you run; not uploaded to GitHub)
  dataset/           downloaded datasets (not uploaded to GitHub)
```

---

## 4. Setup on Windows

Your project folder is `C:\Users\<you>\Downloads\PLCA-`.

### 4.1 First time only
You already have Python 3.12, VS Code and Git. If you ever need to start
from scratch:

```
cd %USERPROFILE%\Downloads
git clone -b claude/great-archimedes-58j5pe https://github.com/riri6967/PLCA-.git
cd PLCA-
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Every time you open a new Command Prompt
```
cd %USERPROFILE%\Downloads\PLCA-
venv\Scripts\activate
```
The line must start with `(venv)`.

### 4.3 Get the latest code (git pull)
```
git pull
pip install -r requirements.txt
python -m unittest
```
`python -m unittest` should end with `OK`.

### 4.4 Disk space
About 2 GB in total: Python, VS Code and Git (~0.9 GB), libraries (~0.5 GB),
ROAD dataset (~0.5 GB zipped, more unzipped), results (< 50 MB).

---

## 5. The phases, step by step

| Phase | What | Script | Status |
|---|---|---|---|
| 1 | Literature review | (you) | to do / in progress |
| 2 | 10BASE-T1S PLCA network model and traffic scenarios | `plca_sim/` | done |
| 3 | Conventional PLCA baseline | `run_baseline.py` | done |
| 4 | TinyML network-state prediction | `run_tinyml.py` | done |
| 5 | Adaptive policy engine + safety governor | `plca_sim/adaptive.py` | done |
| 6 | Conventional vs adaptive comparison | `run_adaptive.py` | done (simulation) |
| 7 | Real traffic (ROAD dataset) | `run_road.py` | built, needs a run on your PC |
| 8 | Write-up (paper / thesis) | (you + Claude) | to do |

### Phase 1: literature review
Papers you already have in `dataset\`:
* `vehiclesec26-song.pdf`: AutoHack multi-bus CAN dataset (USENIX
  VehicleSec 2026).
* `2308.04972v1.pdf`: can-train-and-test CAN dataset (Lampe and Meng).

Also read and cite: the IEEE 802.3cg standard (clause 148, PLCA), the ROAD
dataset paper (Verma et al., PLoS ONE 2024, doi 10.1371/journal.pone.0296879),
PLCA performance and worst-case latency papers, and TinyML surveys. These
fill section 8 of the poster ("Existing research works").

### Phase 2: network model (done)
* 1 coordinator plus N nodes (default 8) on a shared 10 Mb/s bus.
* Model values: 1 bit time = 100 ns, BEACON = 20 bit times, empty TO = 32
  bit times, frame bus time = (frame bytes + 8 preamble) x 8 + 96 IPG.
* Inside a node, critical frames go before non-critical ones.
* Scenarios (offered load = share of the 10 Mb/s bus):
  * **uniform**: all nodes equal (poster scenarios 1-2)
  * **heterogeneous**: a few ECUs carry most traffic
  * **bursty**: frames arrive in bursts (poster scenario 3)
  * **mixed**: 2 ECUs send a 64-byte safety message every 1 ms with a 2 ms
    deadline, other ECUs send large background frames (poster scenario 4)
  * **dynamic**: load and burstiness change every 20-100 ms (driving
    situations)

### Phase 3: conventional baseline (done)
```
python run_baseline.py
```
Output: `results\baseline_overview.png`, `baseline_mixed_criticality.png`,
`baseline_results.csv`, `baseline_summary.csv`.

Key results (average of 3 runs):

| Load | Avg latency uniform | Avg latency bursty | Critical deadline misses (mixed) |
|---|---|---|---|
| 20 % | 0.46 ms | 2.6 ms | 0.15 % |
| 50 % | 0.74 ms | 4.1 ms | 16 % |
| 70 % | 1.38 ms | 6.3 ms | 44 % |
| 90 % | 4.28 ms | 12.9 ms | 66 % |

* Delay rises sharply above ~50 % load.
* Bursty traffic is ~5x worse than smooth traffic even at 20 % load.
* At 90 % load, safety messages are delayed **more** than non-critical ones
  (26 ms vs 11.6 ms): PLCA does not know what is urgent.
* Throughput is fine (up to ~8.7 Mb/s), so the problem is **who is served
  when**, not bus speed.

### Phase 4: TinyML predictor (done)
```
python run_tinyml.py
```
* Every 5 ms the monitor records 15 cheap features: bus and TO utilization,
  arrival rate, queue lengths, critical frames waiting, cycle length, trends
  and smoothed values.
* Each window is labelled with the state of the **next 20 ms**: low
  (< 30 % bus use), medium (< 60 %), high (< 85 %), congested (>= 85 % or
  >= 10 frames waiting).
* A decision tree (depth 4, 31 nodes) is trained on random seeds 0-3 and
  tested on unseen seeds.
* Exported to C: `results\plca_state_model.h`, at most 4 comparisons,
  ~240 bytes of code, so it really fits on an ECU.

Results on unseen runs:

| Predictor | Accuracy | Macro F1 |
|---|---|---|
| TinyML tree (31 nodes) | 62 % | 0.58 |
| "Next = now" persistence | 61 % | 0.55 |
| Much larger gradient boosting | 63 % | 0.60 |
| Oracle that knows the true load | 67 % | - |

* 88 % of predictions are correct or one class off.
* Short-term traffic is random, so no model gets much above ~67 %.
* The tiny model is almost as good as a large one (good TinyML result), but
  predictions cannot be trusted blindly. That is why the safety governor
  exists.

Output: `tinyml_confusion_matrix.png`, `tinyml_timeline.png`,
`tinyml_size_vs_accuracy.png`, `tinyml_per_scenario.png`,
`tinyml_feature_importance.png`, `tinyml_metrics.csv`,
`tinyml_model.joblib`, `plca_state_model.h`.

### Phase 5: adaptive policy engine + safety governor (done)
* **Policy engine:** low or medium state means normal PLCA. High state:
  nodes with >= 3 queued frames may send 2 per TO. Congested state: nodes
  with >= 2 queued frames may send up to 8 per TO.
* **Safety governor:** checks every plan before it reaches the bus and
  never trusts the prediction.
  * For each known safety-critical flow (period P, frame time c, deadline D)
    it fixes `L = ceil(D/P)` critical frames per TO.
  * It gives non-critical traffic a time budget `B`, counted since each
    critical node's last TO. Every node sees the bus, so each node can
    enforce this locally.
  * Proof sketch: the gap between two TOs of a critical node is at most
    `C = beacon + N x TO + sum(L x c) + B`, so every critical frame arrives
    within `C + L x c <= D`. For the mixed scenario the bound is 1.9 ms
    against a 2 ms deadline. The test `tests/test_adaptive.py` checks this.
  * Without critical flows, the governor caps the gap between a node's
    turns at conventional PLCA's own worst case, so adaptation can never
    make the worst case longer.

### Phase 6: conventional vs adaptive (done in simulation)
```
python run_adaptive.py
```
Four policies on the same unseen traffic:
* **conventional**: standard PLCA
* **governor**: governor only, no adaptation
* **reactive**: governor plus adaptation from the measured current state
* **tinyml**: governor plus adaptation from the TinyML prediction (the full
  framework)

Results:

| Result | Conventional | Adaptive (TinyML) |
|---|---|---|
| Critical deadline misses at 50 / 70 / 90 % load | 16 / 42 / 78 % | **0 / 0 / 0 %** |
| Worst critical latency at 90 % load | 218 ms | **1.45 ms** |
| Frame loss at 90 % (mixed) | 15 % | **0.2 %** |
| p99 latency, heterogeneous, 90 % | 24.6 ms | **14.4 ms (-42 %)** |
| p99 latency, dynamic, 90 % | 59 ms | **50 ms (-15 %)** |
| Throughput | - | unchanged |

Honest findings to report:
* The large, guaranteed gain comes from the **safety governor**. Burst
  adaptation gives moderate gains (about 3-23 % lower average latency).
* Non-critical traffic pays the price in the mixed scenario: its average
  latency rises from 18 ms to 33 ms at 90 % load.
* TinyML-driven adaptation is about as good as reactive adaptation. With
  ~62 % prediction accuracy, prediction adds little over measuring.
* With unequal ECU loads, bursts reduce latency but increase jitter for the
  lightly loaded ECUs.

Output: `adaptive_overview.png`, `adaptive_mixed_criticality.png`,
`adaptive_latency_vs_load.png`, `adaptive_results.csv`,
`adaptive_summary.csv`.

### Phase 7: real traffic from the ROAD dataset (needs a run on your PC)
Data location on your PC:
```
dataset\road\road\ambient\*.log      normal driving (USE THESE)
dataset\road\road\attacks\*.log      attacks (do NOT use for this project)
dataset\road\road\signal_extractions decoded signals (not needed)
dataset\road\__MACOSX                Mac leftovers (ignore)
```
Run:
```
python run_road.py
python run_road.py --log "dataset\road\road\ambient\ambient_dyno_drive_winter.log"
```
What it does:
1. Reads the real CAN log, with every message at its real timestamp.
2. Each CAN message becomes a 64-byte Ethernet frame on 10BASE-T1S.
3. CAN IDs are spread over 8 nodes by load (ROAD's DBC is anonymized, so
   the sending ECU is unknown; state this assumption in the paper).
4. The 2 fastest IDs are treated as safety-critical (2 ms deadline).
5. The log is replayed k times with random offsets: a zone that collects k
   CAN buses.
6. Experiment A: CAN only (1-8 CAN buses). Experiment B: CAN plus 30 %
   bulk Ethernet with large frames, such as diagnostics or software
   updates (1-4 CAN buses).

Expected (from a test file in the same format; to be confirmed on real
data): CAN-only traffic is easy for PLCA because the frames are tiny. The
problems appear when large Ethernet frames share the zone, which is where
the governor is needed.

Output: `road_comparison.png`, `road_summary.csv`, `road_results.csv`,
`road_id_stats.csv`.

If it says "Could not read any CAN frames", it prints the first lines of
the log. Give those to Claude.

---

## 6. Datasets

| Dataset | Status | Use |
|---|---|---|
| ROAD | downloaded, unzipped | main real-traffic scenario (ambient logs only) |
| AutoHack (VehicleSec 2026) | paper only | body bus is close to a zonal segment; data link is in the paper |
| can-train-and-test | paper only | 4 cars, to test TinyML generalisation across vehicles |
| CarDS | site not reachable | optional; has Automotive Ethernet (probably 100/1000BASE-T1, not T1S) |
| comma2k19 | skipped | ~100 GB, mostly video |

No public 10BASE-T1S traffic dataset exists yet. Using real CAN traces
migrated to 10BASE-T1S is the standard approach; say so in the paper.

---

## 7. Next steps

1. Run `python run_road.py` on the real ROAD logs (all ambient logs if
   possible) and compare with the simulation results.
2. Check that experiment B on real data still shows conventional PLCA
   missing deadlines while the governor gives 0 %.
3. Optional improvements:
   * train TinyML on ROAD-derived windows as well as synthetic ones
   * reduce the cost to non-critical traffic, for example by adapting the
     non-critical budget to the predicted state
   * add AutoHack / can-train-and-test traces for more vehicles
4. Write the paper: background (section 2 of this guide), model (Phase 2),
   baseline (Phase 3), framework (Phases 4-5), evaluation (Phases 6-7),
   limitations (honest findings above), future work (hardware with
   10BASE-T1S evaluation boards).

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `venv\Scripts\activate`: "The system cannot find the path specified" | You are in the wrong folder. `cd %USERPROFILE%\Downloads\PLCA-` first. If it still fails, run `python -m venv venv` then `pip install -r requirements.txt`. |
| `tar: could not chdir to 'dataset\road'` | Create the folder first: `mkdir dataset\road` |
| `git pull`: "Your local changes would be overwritten" | Run `git stash`, then `git pull`. |
| `ModuleNotFoundError: No module named 'sklearn'` | `pip install -r requirements.txt` with `(venv)` active. |
| A plot or CSV is missing | Run the script for that phase again; `results\` is created automatically. |
| CarDS site gives `ERR_HTTP2_PROTOCOL_ERROR` | InPrivate window, another browser, turn off ad blockers or antivirus HTTPS scanning, or another network. CarDS is optional. |

---

## 9. Glossary

* **10BASE-T1S**: 10 Mb/s single-pair Ethernet with many nodes on one cable.
* **PLCA**: the round-robin access method that prevents collisions on
  10BASE-T1S.
* **BEACON**: the coordinator's signal that starts a PLCA cycle.
* **TO (Transmit Opportunity)**: one node's turn in the cycle.
* **Burst mode**: sending several frames in one TO.
* **ECU**: electronic control unit (a small computer in the car).
* **Latency**: time from a frame being ready to it finishing transmission.
* **Jitter**: how much latency varies.
* **p99**: the latency that 99 % of frames stay below.
* **Deterministic / bounded**: a worst case that is guaranteed, not just
  typical.
* **TinyML**: machine learning small enough to run on microcontrollers.
* **Mixed criticality**: safety-critical and non-critical traffic sharing
  one network.
