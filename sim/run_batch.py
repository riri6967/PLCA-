"""Run an OMNeT++ config in parallel, skipping every run that already finished.

    python3 sim/run_batch.py -f sim/omnetpp.ini -c MixedCriticality
    python3 sim/run_batch.py -f sim/omnetpp.ini -c MixedCriticality --pilot

--pilot runs one short simulation and prints the time and disk the whole config
would need, so a long batch is never started blind. A stopped batch can simply
be started again: finished runs are recorded in results/<config>.done.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RUN_GB = 0.7  # rough memory per simulation; WSL has little, so this sets the limit


def workers() -> int:
    with open("/proc/meminfo") as fh:
        ram_gb = int(fh.readline().split()[1]) / 1048576
    return max(1, min(int(ram_gb / RUN_GB), (os.cpu_count() or 2) - 1))


def base_cmd(args) -> list[str]:
    cmd = ["opp_run_release", "-m", "-u", "Cmdenv", "-f", args.file, "-c", args.config]
    return cmd + ["-n", args.ned] if args.ned else cmd


def num_runs(args) -> int:
    out = subprocess.run(base_cmd(args) + ["-s", "-q", "numruns"], capture_output=True, text=True, check=True)
    return int(re.findall(r"\d+", out.stdout)[-1])


def run_one(args, run: int, extra: list[str] | None = None) -> tuple[int, int, float]:
    t0 = time.time()
    proc = subprocess.run(base_cmd(args) + ["-r", str(run)] + (extra or []), capture_output=True, text=True)
    return run, proc.returncode, time.time() - t0


def pilot(args, total: int) -> None:
    results = Path("results-pilot")
    _, code, seconds = run_one(args, 0, ["--sim-time-limit=6s", f"--result-dir={results}"])
    if code:
        raise SystemExit("pilot run failed")
    size = sum(f.stat().st_size for f in results.rglob("*") if f.is_file())
    factor = 60 / 6
    print(f"{total} runs, {workers()} at once")
    print(f"estimated: {seconds * factor * total / workers() / 3600:.1f} hours, "
          f"{size * factor * total / 1e9:.2f} GB at most")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-f", "--file", default="sim/omnetpp.ini")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-n", "--ned", default=os.environ.get("PLCA_NED", ""), help="NED path")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    total = num_runs(args)
    if args.pilot:
        pilot(args, total)
        return

    done_file = Path("results") / f"{args.config}.done"
    done_file.parent.mkdir(exist_ok=True)
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    todo = [r for r in range(total) if str(r) not in done]
    print(f"{args.config}: {len(done)} of {total} already done, {len(todo)} to run", flush=True)

    failed = 0
    with ThreadPoolExecutor(workers()) as pool:
        futures = [pool.submit(run_one, args, r) for r in todo]
        for f in as_completed(futures):
            run, code, seconds = f.result()
            if code == 0:
                with done_file.open("a") as fh:
                    fh.write(f"{run}\n")
            else:
                failed += 1
                print(f"run {run} failed (code {code})", flush=True)
    print(f"finished, {failed} failed")


if __name__ == "__main__":
    main()
