#!/usr/bin/env python3
"""Runs the experimental sweep from the methodology and collects one CSV.

For every current field and particle count the serial baseline runs once; the
parallel kernel then runs for every (threads, schedule, layout) combination and
is checked for drift against that baseline's final positions.

    python3 scripts/run_sweep.py                       # full sweep on the double gyre
    python3 scripts/run_sweep.py --quick               # a few minutes, for a first look
    python3 scripts/run_sweep.py --fields dg:0.15 nc:data/gulf_stream_hycom.nc

Field specs:  dg:EPS   double gyre with perturbation EPS
              nc:PATH  classic NetCDF file from scripts/prep_currents.py
"""
import argparse
import itertools
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", default=["dg:0.05", "dg:0.15", "dg:0.30"])
    ap.add_argument("--particles", nargs="+", default=["1e4", "1e5", "1e6", "5e6"])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--schedules", nargs="+", default=["static", "dynamic", "guided"])
    ap.add_argument("--layouts", nargs="+", default=["soa", "aos"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--dg-duration", default="10", help="double-gyre run length (model time)")
    ap.add_argument("--nc-duration", default="30", help="NetCDF run length (days)")
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "bench.csv"))
    ap.add_argument("--mpd", default=os.path.join(ROOT, "build", "mpd"))
    ap.add_argument("--quick", action="store_true", help="small sweep: 1e4/1e5 particles, 1 warm-up, 3 runs")
    ap.add_argument("--perf", action="store_true",
                    help="Linux only: wrap each run in `perf stat` and save cache-miss counters next to the CSV")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra arguments passed to mpd")
    args = ap.parse_args()

    if args.quick:
        args.particles, args.warmup, args.runs = ["1e4", "1e5"], 1, 3
    if not os.path.exists(args.mpd):
        sys.exit("%s not found: run `make` first" % args.mpd)
    if args.perf and not shutil.which("perf"):
        sys.exit("--perf needs Linux `perf` on PATH")
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    perf_dir = os.path.splitext(args.csv)[0] + "_perf"
    if args.perf:
        os.makedirs(perf_dir, exist_ok=True)

    # Pin threads to cores where the OpenMP runtime supports it (no effect on macOS).
    env = dict(os.environ, OMP_PROC_BIND="close", OMP_PLACES="cores", OMP_DYNAMIC="false")
    tmp = tempfile.mkdtemp(prefix="mpd_sweep_")
    common = ["--warmup", str(args.warmup), "--runs", str(args.runs), "--csv", args.csv, "--traj", "0"]

    def run(tag, cmd):
        full = [args.mpd] + cmd + common + args.extra
        if args.perf:
            full = ["perf", "stat", "-o", os.path.join(perf_dir, tag + ".txt"), "-e",
                    "cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses"] + full
        print("==", tag, flush=True)
        if subprocess.run(full, env=env).returncode != 0:
            sys.exit("run failed: " + " ".join(full))

    try:
        for spec, p in itertools.product(args.fields, args.particles):
            kind, _, value = spec.partition(":")
            if kind == "dg":
                field = ["--field", "dg", "--eps", value, "--duration", args.dg_duration]
                name = "dg%s" % value
            elif kind == "nc":
                field = ["--field", "nc", "--file", value, "--duration", args.nc_duration]
                name = os.path.splitext(os.path.basename(value))[0]
            else:
                sys.exit("bad field spec '%s' (use dg:EPS or nc:PATH)" % spec)
            base = field + ["--particles", p]

            # The baseline is run per layout so each parallel run is compared like for like.
            for layout in args.layouts:
                ref = os.path.join(tmp, "%s_%s_%s" % (name, p, layout))
                run("%s P=%s %s serial" % (name, p, layout),
                    base + ["--mode", "serial", "--layout", layout, "--out", ref])
                for n, sched in itertools.product(args.threads, args.schedules):
                    run("%s P=%s %s N=%d %s" % (name, p, layout, n, sched),
                        base + ["--mode", "parallel", "--layout", layout, "--threads", str(n),
                                "--schedule", sched, "--ref", ref + "_final.npy"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("results in", args.csv)


if __name__ == "__main__":
    main()
