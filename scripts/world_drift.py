#!/usr/bin/env python3
"""Global drift experiment on OSCAR currents, drawn on Mercator world maps.

Two runs of build/mpd on the same current field and time span:
  * a sparse run: N particles (default 100) scattered evenly over the ocean,
    with their paths recorded -> world_paths.png
  * a dense run: one particle every 1/8 degree of ocean (about 2.5 million),
    so the map of where particles end up is smooth -> world_heatmap.png

    python3 scripts/world_drift.py --field data/oscar_global_2025-12.nc
    python3 scripts/world_drift.py --field data/oscar_global_2025-12.nc --particles 100 --days 30 --out results/world
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
from netCDF4 import Dataset, num2date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EARTH_KM = 6371.0

# Map window: Mercator, split at 20E (through Africa and Europe) so no ocean is cut in half.
SPLIT, LAT_S, LAT_N = 20.0, -70.0, 75.0

# Chart chrome and ramps.
SURFACE, INK, INK2, MUTED, HAIR, LAND = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#b9b8b1"
TIME_RAMP = LinearSegmentedColormap.from_list("elapsed", ["#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "emptier_fuller", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a8a4", "#e34948", "#9c1f24"])
BAR = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- geometry

def merc(lat):
    """Mercator y in degree-equivalent units, so x (degrees) and y share one scale."""
    return np.degrees(np.arcsinh(np.tan(np.radians(lat))))


def wrap(lon):
    """Longitude folded into the map window [SPLIT, SPLIT + 360)."""
    return (np.asarray(lon) - SPLIT) % 360.0 + SPLIT


def haversine_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def lon_label(x):
    lon = (x + 180.0) % 360.0 - 180.0
    if abs(lon) < 1e-9:
        return "0°"
    if abs(abs(lon) - 180.0) < 1e-9:
        return "180°"
    return "%g°%s" % (abs(lon), "E" if lon > 0 else "W")


def lat_label(lat):
    return "0°" if lat == 0 else "%g°%s" % (abs(lat), "N" if lat > 0 else "S")


# ---------------------------------------------------------------- field

class Field:
    """Grid geometry and land mask of a prepared current file."""

    def __init__(self, path):
        with Dataset(path) as ds:
            self.lon = np.asarray(ds["lon"][:], float)
            self.lat = np.asarray(ds["lat"][:], float)
            t = ds["time"]
            self.nt = t.size
            self.start = num2date(float(t[0]), t.units)
            self.span_days = float(t[-1] - t[0]) / 86400.0
            land = np.zeros((self.lat.size, self.lon.size), bool)
            for k in range(self.nt):
                land |= np.isnan(np.ma.filled(ds["u"][k], np.nan)) | np.isnan(np.ma.filled(ds["v"][k], np.nan))
        self.land = land
        self.dx = self.lon[1] - self.lon[0]
        self.dy = self.lat[1] - self.lat[0]
        if abs(self.lon.size * self.dx - 360.0) > 0.5 * self.dx:
            sys.exit("world_drift.py needs a global field (prep_currents.py --region global)")
        # Land grown by two cells: release points keep clear of the coast.
        near = land.copy()
        for s in (1, 2):
            near |= np.roll(land, s, 1) | np.roll(land, -s, 1)
            near[s:] |= land[:-s]
            near[:-s] |= land[s:]
        self.near_land = near

    def wet(self, lon, lat, mask):
        """True where all four grid nodes around (lon, lat) are clear in `mask`."""
        fx = ((lon - self.lon[0]) / self.dx) % self.lon.size
        fy = (lat - self.lat[0]) / self.dy
        i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        i1 = (i0 + 1) % self.lon.size
        ok = (j0 >= 0) & (j0 + 1 < self.lat.size)
        j0c, j1c = np.clip(j0, 0, self.lat.size - 1), np.clip(j0 + 1, 0, self.lat.size - 1)
        return ok & ~(mask[j0c, i0] | mask[j0c, i1] | mask[j1c, i0] | mask[j1c, i1])


def scattered_release(field, count, seed):
    """`count` ocean points, uniform by area and kept apart from each other, so
    they read as evenly scattered rather than clumped. Same seed, same points."""
    rng = np.random.default_rng(seed)
    lo, hi = np.sin(np.radians(LAT_S + 8)), np.sin(np.radians(LAT_N - 8))
    sep = 16.0  # degrees of arc; relaxed until `count` points fit
    while True:
        pts = []
        for _ in range(40000):
            lon = rng.uniform(0.0, 360.0)
            lat = np.degrees(np.arcsin(rng.uniform(lo, hi)))
            if not field.wet(np.array([lon]), np.array([lat]), field.near_land)[0]:
                continue
            if pts:
                q = np.array(pts)
                if haversine_km(q[:, 0], q[:, 1], lon, lat).min() < np.radians(sep) * EARTH_KM:
                    continue
            pts.append((lon, lat))
            if len(pts) == count:
                return np.array(pts), sep
        sep *= 0.9


def dense_release(field, spacing):
    """One particle per `spacing` degrees over all open water in the map window."""
    lon = field.lon[0] + (np.arange(int(round(360.0 / spacing))) + 0.5) * spacing
    lat = LAT_S + (np.arange(int(round((LAT_N - LAT_S) / spacing))) + 0.5) * spacing
    lo, la = np.meshgrid(lon, lat)
    keep = field.wet(lo.ravel(), la.ravel(), field.land)
    return np.column_stack([lo.ravel()[keep], la.ravel()[keep]])


# ---------------------------------------------------------------- running

def run_mpd(args, field_path, release, prefix, extra):
    cmd = [args.mpd, "--field", "nc", "--file", field_path, "--release-file", release,
           "--duration", str(args.days), "--dt", str(args.dt), "--out", prefix] + extra
    print("$", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit("mpd failed")
    with open(prefix + "_meta.json") as f:
        return json.load(f)


# ---------------------------------------------------------------- drawing

def base_map(ax, field):
    """Land, graticule and Mercator axes shared by both figures."""
    order = np.argsort(wrap(field.lon))
    xc = wrap(field.lon)[order]
    xe = np.concatenate([xc - field.dx / 2, [xc[-1] + field.dx / 2]])
    ye = merc(np.clip(np.concatenate([field.lat - field.dy / 2, [field.lat[-1] + field.dy / 2]]), -89.9, 89.9))
    land = np.where(field.land[:, order], 1.0, np.nan)
    ax.pcolormesh(xe, ye, land, cmap=ListedColormap([LAND]), vmin=0, vmax=1, zorder=3, rasterized=True)

    lons = np.arange(30.0, 361.0, 30.0)
    lats = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    for x in lons:
        ax.axvline(x, color=HAIR, linewidth=0.6, zorder=1)
    for y in lats:
        ax.axhline(merc(y), color=HAIR, linewidth=0.6, zorder=1)
    ax.set_xlim(SPLIT, SPLIT + 360.0)
    ax.set_ylim(merc(LAT_S), merc(LAT_N))
    ax.set_aspect("equal")
    ax.set_xticks(lons)
    ax.set_xticklabels([lon_label(x) for x in lons])
    ax.set_yticks(merc(lats))
    ax.set_yticklabels([lat_label(y) for y in lats])
    ax.tick_params(length=0, labelsize=9)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def map_legend(ax):
    """Legend placed over northern Asia, where there is no ocean to cover."""
    ax.legend(loc="upper left", bbox_to_anchor=(0.085, 0.80), fontsize=9, frameon=True, facecolor=SURFACE,
              edgecolor=HAIR, framealpha=0.95, borderpad=0.7).set_zorder(9)


def header(fig, title, subtitle):
    fig.text(0.045, 0.965, title, fontsize=17, fontweight="bold", va="top")
    fig.text(0.045, 0.925, subtitle, fontsize=10.5, color=INK2, va="top", linespacing=1.5)


def stat_block(fig, x, y, rows):
    """A column of big-number / caption pairs."""
    for k, (value, caption) in enumerate(rows):
        fig.text(x, y - 0.034 * k, value, fontsize=14, fontweight="bold", va="top")
        fig.text(x + 0.085, y - 0.034 * k - 0.004, caption, fontsize=9.5, color=INK2, va="top")


def histogram(ax, values, xlabel, ylabel, upper, markers=None):
    edges = np.linspace(0.0, upper, 31)
    counts, _ = np.histogram(np.clip(values, 0, upper), bins=edges)
    ax.bar(edges[:-1], counts, width=np.diff(edges) * 0.86, align="edge", color=BAR, linewidth=0)
    med = float(np.median(values))
    ax.axvline(med, color=INK, linewidth=1)
    ax.annotate("median %s km" % format(int(round(med)), ","), (med, 0.97), xycoords=("data", "axes fraction"),
                xytext=(5, 0), textcoords="offset points", fontsize=9, va="top")
    if markers is not None:
        ax.plot(np.clip(markers, 0, upper), np.full(markers.size, -0.035), "|", color=INK, markersize=7,
                transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlim(0, upper)
    ax.set_xlabel(xlabel + ("" if values.max() <= upper else "  (last bar: %s and beyond)" % format(int(upper), ",")),
                  fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(axis="y", color=HAIR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: format(int(v), ",")))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: format(int(v), ",")))


def period_text(field, meta):
    days = meta["dt"] * meta["steps"] / 86400.0
    first = field.start.strftime("%-d %b %Y")
    if field.nt == 1:
        return days, "%g days of drift on the currents of %s, held steady (only one day of data loaded)" % (days, first)
    last = (field.start + __import__("datetime").timedelta(days=days)).strftime("%-d %b %Y")
    return days, "%g days of drift, %s to %s, on daily currents" % (days, first, last)


def paths_figure(args, field, meta, out):
    traj = np.load(args.out + "/sparse_traj.npy").astype(float)        # [snapshot, particle, (lon, lat)]
    state = np.load(args.out + "/sparse_state.npy")
    n = traj.shape[1]
    days, period = period_text(field, meta)
    hours = np.arange(traj.shape[0]) * meta["traj_every"] * meta["dt"] / 3600.0

    step_km = haversine_km(traj[:-1, :, 0], traj[:-1, :, 1], traj[1:, :, 0], traj[1:, :, 1])
    path_km = step_km.sum(axis=0)
    net_km = haversine_km(traj[0, :, 0], traj[0, :, 1], traj[-1, :, 0], traj[-1, :, 1])

    fig = plt.figure(figsize=(15, 12.4))
    header(fig, "Where %d floating particles drift in %g days" % (n, days),
           "%s.\nOSCAR surface currents, 0.25° global grid (%d × %d). Each particle is a passive tracer advanced "
           "hourly with RK4." % (period, field.lon.size, field.lat.size))
    ax = fig.add_axes([0.045, 0.265, 0.875, 0.60])
    base_map(ax, field)

    x, y = wrap(traj[:, :, 0]), merc(traj[:, :, 1])
    segs, colour = [], []
    for k in range(n):
        pts = np.column_stack([x[:, k], y[:, k]])
        pair = np.stack([pts[:-1], pts[1:]], axis=1)
        ok = np.abs(pair[:, 1, 0] - pair[:, 0, 0]) < 180.0   # drop the jump across the map edge
        segs.append(pair[ok])
        colour.append(0.5 * (hours[:-1] + hours[1:])[ok] / 24.0)
    lc = LineCollection(np.concatenate(segs), array=np.concatenate(colour), cmap=TIME_RAMP, linewidths=2.0,
                        capstyle="round", zorder=5)
    lc.set_clim(0, days)
    ax.add_collection(lc)

    afloat = state == 0
    ax.scatter(x[0], y[0], s=30, facecolor=SURFACE, edgecolor=INK, linewidth=1.1, zorder=6, label="release point")
    ax.scatter(x[-1, afloat], y[-1, afloat], s=26, facecolor="#0d366b", edgecolor=SURFACE, linewidth=0.9, zorder=7,
               label="position after %g days" % days)
    if (~afloat).any():
        ax.scatter(x[-1, ~afloat], y[-1, ~afloat], s=44, marker="X", facecolor=INK, edgecolor=SURFACE, linewidth=0.8,
                   zorder=7, label="stopped: ran aground or left the grid")
    # Name the distance for the five particles that travelled farthest.
    for k in np.argsort(path_km)[-5:]:
        ax.annotate("%s km" % format(int(round(path_km[k], -1)), ","), (x[-1, k], y[-1, k]), xytext=(7, 6),
                    textcoords="offset points", fontsize=8.5, color=INK, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85))
    map_legend(ax)

    cax = fig.add_axes([0.935, 0.34, 0.012, 0.45])
    cb = fig.colorbar(lc, cax=cax)
    cb.set_label("days since release (path colour)", fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)

    hax = fig.add_axes([0.075, 0.055, 0.42, 0.15])
    histogram(hax, path_km, "distance travelled along the path (km)", "particles",
              upper=float(np.ceil(np.percentile(path_km, 97) / 100.0) * 100.0))
    fig.text(0.075, 0.222, "How far the %d particles went" % n, fontsize=11, fontweight="bold")

    fig.text(0.56, 0.222, "In numbers", fontsize=11, fontweight="bold")
    stat_block(fig, 0.56, 0.192, [
        ("%s km" % format(int(round(np.median(path_km))), ","), "median distance travelled along the path"),
        ("%s km" % format(int(round(path_km.max(), -1)), ","), "farthest traveller (labelled on the map with the next four)"),
        ("%s km" % format(int(round(np.median(net_km))), ","), "median straight-line distance from release to end"),
        ("%.2f m/s" % (np.median(path_km) * 1000.0 / (days * 86400.0)), "median average speed"),
        ("%d / %d" % (afloat.sum(), n), "still afloat; %d ran aground, %d left the grid" % (meta["beached"], meta["out"])),
    ])
    fig.text(0.045, 0.012, "Map: Mercator, %g°S to %g°N, split at %g°E. Land is the OSCAR land/ice mask. Release points are "
             "random, uniform by ocean area, at least %.0f° apart and clear of the coast (seed %d)."
             % (-LAT_S, LAT_N, SPLIT, args.sep, args.seed), fontsize=8.5, color=MUTED)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def heatmap_figure(args, field, meta, timing, out):
    final = np.load(args.out + "/dense_final.npy")
    release = np.load(args.out + "/dense_release.npy")
    days, period = period_text(field, meta)
    n = final.shape[0]

    # One-degree boxes. Colour is the count at the end over the count at release in the same box.
    xe = np.arange(SPLIT, SPLIT + 360.0 + 1e-9, args.bin)
    le = np.arange(LAT_S, LAT_N + 1e-9, args.bin)
    start, _, _ = np.histogram2d(wrap(release[:, 0]), release[:, 1], bins=[xe, le])
    end, _, _ = np.histogram2d(wrap(final[:, 0]), final[:, 1], bins=[xe, le])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(start > 0, end / start, np.nan)
    spread = np.abs(np.log(ratio[np.isfinite(ratio) & (ratio > 0)]))
    top = float(np.clip(np.exp(np.percentile(spread, 99.0)), 1.5, 4.0))
    top = np.round(top * 2) / 2
    shown = np.clip(ratio, 1.0 / top, top)   # emptied boxes take the darkest "emptier" colour

    fig = plt.figure(figsize=(15, 12.4))
    header(fig, "Where floating particles pile up after %g days" % days,
           "%s.\n%s particles, one released every %g° of open ocean, counted in %g° boxes at the end. "
           "OSCAR surface currents, 0.25° global grid." % (period, format(n, ","), args.spacing, args.bin))
    ax = fig.add_axes([0.045, 0.265, 0.875, 0.60])
    base_map(ax, field)
    im = ax.pcolormesh(xe, merc(le), shown.T, cmap=DIVERGING, norm=LogNorm(vmin=1.0 / top, vmax=top), zorder=2,
                       rasterized=True)

    sparse = args.out + "/sparse_final.npy"
    if os.path.exists(sparse):
        s = np.load(sparse)
        ax.scatter(wrap(s[:, 0]), merc(s[:, 1]), s=13, facecolor=INK, edgecolor=SURFACE, linewidth=0.7, zorder=6,
                   label="end points of the %d tracked particles (see world_paths.png)" % s.shape[0])
        map_legend(ax)

    cax = fig.add_axes([0.935, 0.34, 0.012, 0.45])
    cb = fig.colorbar(im, cax=cax)
    ticks = [t for t in (0.25, 1 / 3, 0.5, 1.0, 2.0, 3.0, 4.0) if 1.0 / top - 1e-9 <= t <= top + 1e-9]
    cb.set_ticks(ticks)
    cb.set_ticklabels(["same" if t == 1 else ("%g×" % t if t > 1 else "1/%d" % round(1 / t)) for t in ticks])
    cb.ax.minorticks_off()
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    cb.set_label("particles in the box at the end ÷ at release", fontsize=9.5)
    fig.text(0.941, 0.80, "fuller", fontsize=9, color=INK2, ha="center")
    fig.text(0.941, 0.322, "emptier", fontsize=9, color=INK2, ha="center")

    net_km = haversine_km(release[:, 0], release[:, 1], final[:, 0], final[:, 1])
    hax = fig.add_axes([0.075, 0.055, 0.42, 0.15])
    histogram(hax, net_km, "straight-line distance from release to end (km)", "particles",
              upper=float(np.ceil(np.percentile(net_km, 99) / 100.0) * 100.0))
    fig.text(0.075, 0.222, "How far particles ended from where they started", fontsize=11, fontweight="bold")

    fig.text(0.56, 0.222, "In numbers", fontsize=11, fontweight="bold")
    finite = ratio[np.isfinite(ratio)]
    rows = [
        ("%.0f%%" % (100.0 * np.mean(finite > 1.25)), "of ocean boxes gained more than a quarter (red)"),
        ("%.0f%%" % (100.0 * np.mean(finite < 0.8)), "of ocean boxes lost more than a fifth (blue)"),
        ("%.1f%%" % (100.0 * meta["beached"] / n), "of particles ran aground (%s)" % format(meta["beached"], ",")),
    ]
    if timing.get("parallel"):
        rows.append(("%.1f s" % timing["parallel"],
                     "to advance %s particles %s steps on %d threads" % (format(n, ","), format(meta["steps"], ","), timing["threads"])))
    if timing.get("serial"):
        rows.append(("%.1f×" % (timing["serial"] / timing["parallel"]),
                     "faster than the serial run (%.0f s); largest position difference %s km"
                     % (timing["serial"], "%g" % timing["drift"])))
    stat_block(fig, 0.56, 0.192, rows)
    fig.text(0.045, 0.012, "Map: Mercator, %g°S to %g°N, split at %g°E. Colour scale is logarithmic and clipped at %g×; "
             "boxes with no particle released in them are left blank. Grey is the OSCAR land/ice mask."
             % (-LAT_S, LAT_N, SPLIT, top), fontsize=8.5, color=MUTED)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True, help="global classic NetCDF from prep_currents.py --region global")
    ap.add_argument("--particles", type=int, default=100, help="tracked particles in the sparse run")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=3600.0, help="timestep in seconds")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--spacing", type=float, default=0.125, help="dense release spacing in degrees")
    ap.add_argument("--bin", type=float, default=1.0, help="heatmap box size in degrees")
    ap.add_argument("--threads", type=int, default=0, help="threads for the dense run [all]")
    ap.add_argument("--no-serial", action="store_true", help="skip the serial baseline of the dense run")
    ap.add_argument("--plot-only", action="store_true", help="redraw from existing outputs")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "world"))
    ap.add_argument("--mpd", default=os.path.join(ROOT, "build", "mpd"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    field = Field(args.field)
    runs_csv = os.path.join(args.out, "runs.csv")
    setup = os.path.join(args.out, "setup.json")

    if not args.plot_only:
        pts, args.sep = scattered_release(field, args.particles, args.seed)
        np.save(os.path.join(args.out, "release_sparse.npy"), pts)
        dense = dense_release(field, args.spacing)
        np.save(os.path.join(args.out, "release_dense.npy"), dense)
        print("release: %d scattered points (at least %.1f deg apart), %s dense points" % (len(pts), args.sep, format(len(dense), ",")))
        with open(setup, "w") as f:
            json.dump({"sep": args.sep, "seed": args.seed, "spacing": args.spacing}, f)
        if os.path.exists(runs_csv):
            os.remove(runs_csv)
        every = max(1, int(round(6 * 3600.0 / args.dt)))  # a path sample every six hours
        run_mpd(args, args.field, os.path.join(args.out, "release_sparse.npy"), os.path.join(args.out, "sparse"),
                ["--mode", "serial", "--runs", "1", "--warmup", "0", "--traj", str(args.particles),
                 "--traj-every", str(every)])
        extra = ["--mode", "parallel" if args.no_serial else "both", "--schedule", "guided", "--runs", "1",
                 "--warmup", "0", "--traj", "0", "--csv", runs_csv]
        if args.threads:
            extra += ["--threads", str(args.threads)]
        run_mpd(args, args.field, os.path.join(args.out, "release_dense.npy"), os.path.join(args.out, "dense"), extra)

    with open(setup) as f:
        s = json.load(f)
    args.sep, args.seed, args.spacing = s["sep"], s["seed"], s["spacing"]
    timing = {}
    if os.path.exists(runs_csv):
        with open(runs_csv) as f:
            for r in csv.DictReader(f):
                timing[r["mode"]] = float(r["t_median"])
                if r["mode"] == "parallel":
                    timing["threads"] = int(r["threads"])
                    timing["drift"] = float(r["drift_max"]) if r["drift_max"] else float("nan")
    with open(os.path.join(args.out, "sparse_meta.json")) as f:
        sparse_meta = json.load(f)
    with open(os.path.join(args.out, "dense_meta.json")) as f:
        dense_meta = json.load(f)
    paths_figure(args, field, sparse_meta, os.path.join(args.out, "world_paths.png"))
    heatmap_figure(args, field, dense_meta, timing, os.path.join(args.out, "world_heatmap.png"))


if __name__ == "__main__":
    main()
