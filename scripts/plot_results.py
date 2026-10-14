#!/usr/bin/env python3
"""Figures for the report.

Scaling figures from a sweep CSV (speedup, efficiency, schedule/layout balance):
    python3 scripts/plot_results.py scaling results/bench.csv --outdir results/figs

Dispersion map from one run's outputs (mpd --out PREFIX):
    python3 scripts/plot_results.py map results/gulf --png results/figs/gulf_map.png
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Validated categorical palette (colour-blind safe ordering), light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87fa7", "#008300", "#4a3aa7", "#e52f3c"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dc"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
})


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("eps", "particles", "threads", "t_median", "throughput", "busy_cv", "busy_max_over_mean",
                  "drift_max"):
            r[k] = float(r[k]) if r[k] != "" else float("nan")
        r["threads"] = int(r["threads"])
        r["particles"] = int(r["particles"])
        r["label"] = "double gyre, ε=%g" % r["eps"] if r["field"] == "double_gyre" else r["field"].replace(".nc", "")
    return rows


def fmt_p(p):
    exp = int(np.floor(np.log10(p)))
    mant = p / 10 ** exp
    return ("10$^{%d}$" % exp) if abs(mant - 1) < 1e-9 else ("%g×10$^{%d}$" % (mant, exp))


def thread_axis(ax, threads):
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.minorticks_off()
    ax.set_xlabel("threads")


def scaling(args):
    rows = load(args.csv)
    os.makedirs(args.outdir, exist_ok=True)
    serial = {(r["label"], r["particles"], r["layout"]): r["t_median"] for r in rows if r["mode"] == "serial"}
    par = [r for r in rows if r["mode"] == "parallel" and (r["label"], r["particles"], r["layout"]) in serial]
    for r in par:
        r["S"] = serial[(r["label"], r["particles"], r["layout"])] / r["t_median"]
        r["E"] = r["S"] / r["threads"]
    if not par:
        raise SystemExit("no parallel rows with a matching serial baseline in " + args.csv)
    made = []

    # RQ1: speedup and efficiency vs threads, one line per particle count, one figure per field.
    for label in sorted({r["label"] for r in par}):
        sel = [r for r in par if r["label"] == label and r["layout"] == args.layout and r["schedule"] == args.schedule]
        if not sel:
            continue
        threads = sorted({r["threads"] for r in sel})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
        for k, p in enumerate(sorted({r["particles"] for r in sel})):
            pts = sorted((r["threads"], r["S"], r["E"]) for r in sel if r["particles"] == p)
            n, s, e = zip(*pts)
            a1.plot(n, s, "o-", color=SERIES[k % len(SERIES)], label="P = " + fmt_p(p))
            a2.plot(n, [100 * x for x in e], "o-", color=SERIES[k % len(SERIES)], label="P = " + fmt_p(p))
        a1.plot(threads, threads, "--", color=MUTED, linewidth=1, label="ideal")
        a1.set_yscale("log", base=2)
        a1.set_yticks(threads)
        a1.set_yticklabels([str(t) for t in threads])
        a1.set_ylabel("speedup  S(N) = T$_{serial}$ / T$_{parallel}$(N)")
        a1.set_title("Speedup")
        a2.axhline(100, linestyle="--", color=MUTED, linewidth=1)
        a2.set_ylim(0, 115)
        a2.set_ylabel("parallel efficiency  E(N) = S(N) / N  (%)")
        a2.set_title("Efficiency")
        for a in (a1, a2):
            thread_axis(a, threads)
        a1.legend()
        fig.suptitle("%s  ·  %s, schedule(%s)" % (label, args.layout.upper(), args.schedule), fontsize=10, x=0.02, y=1.03, ha="left")
        out = os.path.join(args.outdir, "rq1_scaling_%s.png" % label.replace(" ", "_").replace(",", "").replace("ε=", "eps"))
        fig.savefig(out)
        plt.close(fig)
        made.append(out)

    # RQ3: schedule x layout at the largest particle count and a fixed thread count.
    pmax = max(r["particles"] for r in par)
    nfix = args.threads or max(r["threads"] for r in par)
    for label in sorted({r["label"] for r in par}):
        sel = [r for r in par if r["label"] == label and r["particles"] == pmax and r["threads"] == nfix]
        scheds = [s for s in ("static", "dynamic", "guided") if any(r["schedule"] == s for r in sel)]
        layouts = [l for l in ("soa", "aos") if any(r["layout"] == l for r in sel)]
        if not scheds:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
        metrics = (("busy_cv", "per-thread busy-time CV (std / mean)", "Load imbalance"),
                   ("t_median", "median wall-clock time (s)", "Runtime"),
                   ("S", "speedup over serial", "Speedup"))
        width = 0.8 / max(1, len(layouts))
        for ax, (key, ylabel, title) in zip(axes, metrics):
            for li, layout in enumerate(layouts):
                vals = [next((r[key] for r in sel if r["schedule"] == s and r["layout"] == layout), np.nan) for s in scheds]
                x = np.arange(len(scheds)) + (li - (len(layouts) - 1) / 2) * width
                bars = ax.bar(x, vals, width * 0.92, color=SERIES[li], label=layout.upper())
                ax.bar_label(bars, fmt="%.2g", fontsize=7, color=MUTED, padding=2)
            ax.set_xticks(np.arange(len(scheds)))
            ax.set_xticklabels(scheds)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(bottom=0)
            ax.grid(axis="x", visible=False)
        axes[0].legend()
        fig.suptitle("%s  ·  P = %s, %d threads" % (label, fmt_p(pmax), nfix), fontsize=10, x=0.02, y=1.03, ha="left")
        out = os.path.join(args.outdir, "rq3_schedule_layout_%s.png" % label.replace(" ", "_").replace(",", "").replace("ε=", "eps"))
        fig.savefig(out)
        plt.close(fig)
        made.append(out)

    # RQ2: throughput vs threads across fields (different grid resolutions) at the largest P.
    sel = [r for r in par if r["particles"] == pmax and r["layout"] == args.layout and r["schedule"] == args.schedule]
    if sel:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        threads = sorted({r["threads"] for r in sel})
        for k, label in enumerate(sorted({r["label"] for r in sel})):
            pts = sorted((r["threads"], r["throughput"] / 1e6) for r in sel if r["label"] == label)
            grid = next("%s×%s" % (r["nx"], r["ny"]) for r in sel if r["label"] == label)
            ax.plot(*zip(*pts), "o-", color=SERIES[k % len(SERIES)], label="%s (%s grid)" % (label, grid))
        thread_axis(ax, threads)
        ax.set_yscale("log")
        ax.set_ylabel("throughput (million particle-steps / s)")
        ax.set_title("Kernel throughput, P = %s" % fmt_p(pmax))
        ax.legend(fontsize=7)
        out = os.path.join(args.outdir, "rq2_throughput.png")
        fig.savefig(out)
        plt.close(fig)
        made.append(out)

    # RQ4 is a table, not a chart: worst drift per field.
    print("\nRQ4  worst position drift of any parallel run against its serial baseline")
    for label in sorted({r["label"] for r in par}):
        d = [r["drift_max"] for r in par if r["label"] == label and not np.isnan(r["drift_max"])]
        print("  %-28s %s" % (label, ("%.3g over %d runs" % (max(d), len(d))) if d else "not measured"))
    print("\nwrote:")
    for m in made:
        print("  " + m)


def dispersion_map(args):
    with open(args.prefix + "_meta.json") as f:
        meta = json.load(f)
    final = np.load(args.prefix + "_final.npy")
    state = np.load(args.prefix + "_state.npy")
    release = np.load(args.prefix + "_release.npy")
    x0, x1, y0, y1 = meta["extent"]
    sph = meta["spherical"]
    aspect = (1.0 / np.cos(np.radians(0.5 * (y0 + y1)))) if sph else 1.0
    width = 11.0
    panel_h = (width / 2 - 1.2) * (y1 - y0) / (x1 - x0) * aspect
    fig, axes = plt.subplots(1, 2, figsize=(width, panel_h + 1.3), sharex=True, sharey=True)

    bins = (args.bins, max(8, int(args.bins * (y1 - y0) / (x1 - x0) * aspect)))
    afloat = state == 0
    h_end, xe, ye = np.histogram2d(final[afloat, 0], final[afloat, 1], bins=bins, range=[[x0, x1], [y0, y1]])
    h_start, _, _ = np.histogram2d(release[:, 0], release[:, 1], bins=bins, range=[[x0, x1], [y0, y1]])
    # Density relative to a uniform spread of the same particles: >1 means accumulation.
    uniform = max(h_start.mean(), 1e-12)
    rel = np.where(h_end > 0, h_end / uniform, np.nan)

    land = None
    if os.path.exists(args.prefix + "_land.npy"):
        land = np.load(args.prefix + "_land.npy").astype(float)
        land[land == 0] = np.nan

    def backdrop(ax):
        ax.set_facecolor("#f1f1ec")
        ax.grid(False)
        if land is not None and np.isfinite(land).any():
            ax.imshow(land, origin="lower", extent=[x0, x1, y0, y1], cmap="Greys", vmin=0, vmax=1.6,
                      interpolation="nearest", aspect="auto", zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(aspect)

    ax = axes[0]
    backdrop(ax)
    if os.path.exists(args.prefix + "_traj.npy"):
        traj = np.load(args.prefix + "_traj.npy")
        for k in range(traj.shape[1]):
            seg = traj[:, k, :]
            if sph:  # break lines that wrap around the globe
                seg = np.where(np.abs(np.diff(seg[:, 0], prepend=seg[0, 0]))[:, None] > 180, np.nan, seg)
            ax.plot(seg[:, 0], seg[:, 1], color=SERIES[0], linewidth=0.5, alpha=0.55, zorder=2)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=4, color=INK, zorder=4, linewidths=0)
        ax.set_title("Trajectories of %d tracked particles (dots: release)" % traj.shape[1])
    else:
        ax.set_title("No trajectories recorded (run mpd with --traj)")

    ax = axes[1]
    backdrop(ax)
    vmax = max(4.0, np.nanpercentile(rel, 99.5))
    im = ax.pcolormesh(xe, ye, rel.T, cmap="viridis", norm=LogNorm(vmin=1.0 / vmax, vmax=vmax), zorder=1)
    beached = state == 1
    if beached.any():
        ax.scatter(final[beached, 0], final[beached, 1], s=1.5, color="#e52f3c", zorder=5, linewidths=0,
                   label="beached (%d)" % beached.sum())
        ax.legend(loc="upper right", fontsize=7, markerscale=4, frameon=True, facecolor="white", edgecolor=GRID)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("particle density ÷ uniform density")
    unit = "days" if sph else "time units"
    span = meta["dt"] * meta["steps"] / (86400.0 if sph else 1.0)
    ax.set_title("Accumulation after %.3g %s (%s particles)" % (span, unit, format(meta["particles"], ",")))
    for a in axes:
        a.set_xlabel("longitude (°E)" if sph else "x")
    axes[0].set_ylabel("latitude (°N)" if sph else "y")
    title = meta["field"] if sph else "double gyre, ε = %g" % meta["eps"]
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.06, ha="left")
    out = args.png or args.prefix + "_map.png"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scaling", help="speedup / efficiency / schedule figures from a sweep CSV")
    s.add_argument("csv")
    s.add_argument("--outdir", default="results/figs")
    s.add_argument("--layout", default="soa", help="layout shown in the RQ1/RQ2 figures")
    s.add_argument("--schedule", default="static", help="schedule shown in the RQ1/RQ2 figures")
    s.add_argument("--threads", type=int, default=0, help="thread count for the RQ3 figure [largest]")
    s.set_defaults(func=scaling)
    m = sub.add_parser("map", help="trajectory + accumulation map from one run")
    m.add_argument("prefix")
    m.add_argument("--png")
    m.add_argument("--bins", type=int, default=160)
    m.set_defaults(func=dispersion_map)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
