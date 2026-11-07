"""Run an OMNeT++ config in parallel, skipping every run that already finished.

    source ~/plca/omnetpp-6.4.0/setenv -q
    python3 run_batch.py S4_Mixed
    python3 run_batch.py S4_Mixed --pilot

--pilot runs one short simulation and prints the time and disk the whole config
would need. Each finished run is reduced to one row of results/summary.csv and
logged in results/<config>.done, so a stopped batch can simply be started again.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
INET = Path(os.environ.get("INET_ROOT", Path.home() / "plca/inet-4.7.0"))
# Safety followers and deadline in ms, per config.
SAFETY = {"S4_Mixed": ("0,1", 2.0)}
SAFETY.update({c: ("0,1", 10.0) for c in ("B2_BestStatic", "B3_FixedPriority", "P_Proposed", "S5_B3", "T_Conventional")})
SAFETY.update({c: ("0,1", 20.0) for c in ("B2_Tuned", "B3_20", "B4_20", "P20", "A1_20", "A2_20", "A3_20", "M_20", "W2_20",
                                          "W20_20", "S5_MixedBursty", "S5_B3_20", "S5_B4_20", "S5_P20", "T_Proposed")})
SAFETY.update({f"D{d}_Deadline": ("0,1", float(d)) for d in (15, 30)})


def workers() -> int:
    """Each run needs about 70 MB, so cores set the limit. One stays free."""
    return max(1, (os.cpu_count() or 2) - 2)


def base_cmd(config: str) -> list[str]:
    lib = ["-l", str(HERE / "plcasim")] if (HERE / "libplcasim.so").exists() else []
    return ["opp_run_release", "-m", "-u", "Cmdenv", "-l", str(INET / "src/INET"), *lib,
            "-n", f"{HERE}:{INET / 'src'}", "-f", str(HERE / "omnetpp.ini"), "-c", config]


def num_runs(config: str) -> int:
    out = subprocess.run(base_cmd(config) + ["-q", "numruns"], capture_output=True, text=True, check=True)
    return int(re.findall(r"Number of runs: (\d+)", out.stdout)[-1])


def run_one(config: str, run: int, result_dir: Path, extra: list[str]) -> tuple[int, int, float]:
    t0 = time.time()
    cmd = base_cmd(config) + ["-r", str(run), f"--result-dir={result_dir}"] + extra
    code = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE).returncode
    return run, code, time.time() - t0


def collect(config: str, result_dir: Path, summary: Path) -> None:
    safety, deadline = SAFETY.get(config, ("", 2.0))
    for sca in result_dir.glob("*.sca"):
        subprocess.run([sys.executable, str(HERE / "collect.py"), str(sca), "--out", str(summary),
                        "--safety", safety, "--deadline-ms", str(deadline)],
                       capture_output=True, text=True, check=True)


def pilot(config: str, total: int) -> None:
    probe = HERE / "results" / f"{config}-pilot"
    _, code, seconds = run_one(config, total - 1, probe, ["--sim-time-limit=6s", "--warmup-period=1s"])
    if code:
        raise SystemExit("pilot run failed")
    size = sum(f.stat().st_size for f in probe.rglob("*") if f.is_file())
    factor = 10
    print(f"{config}: {total} runs, {workers()} at once")
    print(f"estimated {seconds * factor * total / workers() / 3600:.1f} hours and "
          f"{size * factor * total / 1e9:.2f} GB before raw files are removed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--extra", default="", help="extra opp_run options, space separated")
    args = ap.parse_args()

    total = num_runs(args.config)
    if args.pilot:
        pilot(args.config, total)
        return

    results = HERE / "results"
    results.mkdir(exist_ok=True)
    done_file = results / f"{args.config}.done"
    summary = results / "summary.csv"
    lock = results / ".lock"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    todo = [r for r in range(total) if str(r) not in done]
    print(f"{args.config}: {len(done)} of {total} done, {len(todo)} to run on {workers()} cores", flush=True)

    def task(run: int) -> tuple[int, int, float]:
        rdir = results / args.config / f"run{run}"
        out = run_one(args.config, run, rdir, args.extra.split())
        if out[1] == 0:
            with lock.open("w") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                collect(args.config, rdir, summary)
                with done_file.open("a") as d:
                    d.write(f"{run}\n")
        return out

    failed = 0
    finished = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers()) as pool:
        for f in as_completed([pool.submit(task, r) for r in todo]):
            run, code, seconds = f.result()
            finished += 1
            if code:
                failed += 1
                print(f"run {run} failed, code {code}", flush=True)
            if finished % 10 == 0 or finished == len(todo):
                left = (time.time() - t0) / finished * (len(todo) - finished) / 60
                print(f"{finished}/{len(todo)} done, about {left:.0f} min left", flush=True)
    print(f"finished, {failed} failed")


if __name__ == "__main__":
    main()
