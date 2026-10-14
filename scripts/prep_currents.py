#!/usr/bin/env python3
"""Subset OSCAR / HYCOM surface currents to one region and write the classic
NetCDF file the C++ loader reads: float32 u, v on (time, lat, lon), NaN on land.

The inputs are NetCDF-4 (or an OPeNDAP URL); this is the one place the netCDF4
library is used, and it runs once, outside anything that is timed.

OSCAR (daily files downloaded from PO.DAAC, needs a NASA Earthdata login):
    python3 scripts/prep_currents.py --region california \
        --out data/california_oscar.nc data/oscar/oscar_currents_nrt_2026*.nc

HYCOM: use scripts/fetch_hycom.sh, which downloads daily files and calls this
script on them (GLBy0.08 is 0.04 deg in latitude, hence --lat-stride 2).
"""
import argparse
import sys
from datetime import datetime

import numpy as np
from netCDF4 import Dataset, date2num, num2date

# lon_min, lon_max, lat_min, lat_max (degrees east in -180..180, degrees north)
REGIONS = {
    "npac_gyre": (-160.0, -130.0, 25.0, 40.0),     # North Pacific Gyre interior: calm
    "california": (-135.0, -115.0, 25.0, 45.0),    # California Current: moderate eddies
    "gulf_stream": (-75.0, -45.0, 30.0, 45.0),     # Gulf Stream Extension: eddy-rich
    "agulhas": (5.0, 40.0, -45.0, -30.0),          # Agulhas Retroflection: eddy-rich
    "global": None,                                # the whole grid, no subsetting
}
U_NAMES = ("u", "water_u", "uo", "ugos")
V_NAMES = ("v", "water_v", "vo", "vgos")
LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "longitude")
TIME_NAMES = ("time",)


def pick(ds, names, what):
    for n in names:
        if n in ds.variables:
            return ds.variables[n]
    sys.exit("no %s variable found (looked for %s)" % (what, ", ".join(names)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="NetCDF files (one or many along time) or one OPeNDAP URL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", choices=sorted(REGIONS))
    ap.add_argument("--bbox", help="lon_min,lon_max,lat_min,lat_max (overrides --region)")
    ap.add_argument("--start", help="first date, YYYY-MM-DD")
    ap.add_argument("--end", help="last date, YYYY-MM-DD")
    ap.add_argument("--time-stride", type=int, default=1)
    ap.add_argument("--lat-stride", type=int, default=1, help="keep every n-th latitude (HYCOM GLBy is 0.04 deg in lat)")
    ap.add_argument("--lon-stride", type=int, default=1)
    ap.add_argument("--depth-index", type=int, default=0, help="level to take if the data has a depth axis")
    args = ap.parse_args()

    box = None
    if args.bbox:
        box = tuple(float(x) for x in args.bbox.split(","))
    elif args.region:
        box = REGIONS[args.region]
    else:
        sys.exit("give --region or --bbox")

    # Several files are stitched along time by hand (MFDataset cannot read NetCDF-4).
    datasets = [Dataset(p) for p in sorted(args.inputs)]
    ds = datasets[0]
    uvar, vvar = pick(ds, U_NAMES, "u"), pick(ds, V_NAMES, "v")
    latv, lonv, timev = pick(ds, LAT_NAMES, "latitude"), pick(ds, LON_NAMES, "longitude"), pick(ds, TIME_NAMES, "time")
    lat, lon = np.asarray(latv[:], float), np.asarray(lonv[:], float)

    if box is None:
        ii, jj = np.arange(lon.size), np.arange(lat.size)
    else:
        # Express the box in the dataset's own longitude convention (0..360 or -180..180).
        lon_min, lon_max, lat_min, lat_max = box
        if lon.max() > 180.0:
            lon_min, lon_max = lon_min % 360.0, lon_max % 360.0
        if lon_min >= lon_max:
            sys.exit("region crosses the dataset's longitude seam; not supported")
        ii = np.where((lon >= lon_min - 1e-6) & (lon <= lon_max + 1e-6))[0]
        jj = np.where((lat >= lat_min - 1e-6) & (lat <= lat_max + 1e-6))[0]
    if ii.size < 2 or jj.size < 2:
        sys.exit("region selects fewer than 2 grid points")
    ii, jj = ii[:: args.lon_stride], jj[:: args.lat_stride]
    isl = slice(ii[0], ii[-1] + 1, args.lon_stride)
    jsl = slice(jj[0], jj[-1] + 1, args.lat_stride)

    # Every (file, index) time slice, in time order.
    slices = []
    for d in datasets:
        tv = pick(d, TIME_NAMES, "time")
        if d is not ds and (pick(d, LAT_NAMES, "latitude").shape != latv.shape or
                            pick(d, LON_NAMES, "longitude").shape != lonv.shape):
            sys.exit("input files are on different grids")
        units, calendar = tv.units, getattr(tv, "calendar", "standard")
        raw = np.atleast_1d(np.asarray(tv[:], float))
        keep = np.ones(raw.size, bool)
        if args.start:
            keep &= raw >= date2num(datetime.strptime(args.start, "%Y-%m-%d"), units, calendar)
        if args.end:
            keep &= raw <= date2num(datetime.strptime(args.end, "%Y-%m-%d"), units, calendar)
        stamps = np.atleast_1d(num2date(raw, units, calendar))  # cftime dates: any calendar works
        slices += [(stamps[k], d, k) for k in np.where(keep)[0]]
    slices.sort(key=lambda s: s[0])
    slices = [s for k, s in enumerate(slices) if k == 0 or s[0] > slices[k - 1][0]]
    slices = slices[:: args.time_stride]
    if not slices:
        sys.exit("no time slices in the selected window")
    if len(slices) == 1:
        print("note: only one time slice; the simulator will treat it as a steady field", file=sys.stderr)

    # Read slice by slice (keeps OPeNDAP requests small), in whatever dimension order the source uses.
    dims = uvar.dimensions
    axis = {d: k for k, d in enumerate(dims)}
    tdim, ydim, xdim = timev.dimensions[0], latv.dimensions[0], lonv.dimensions[0]
    order = [axis[ydim], axis[xdim]]

    def read(d, names, what, t):
        var = pick(d, names, what)
        index = []
        for dim in dims:
            index.append(int(t) if dim == tdim else jsl if dim == ydim else isl if dim == xdim else args.depth_index)
        a = np.ma.filled(np.ma.masked_invalid(var[tuple(index)]).astype(np.float32), np.nan)
        return a if order[0] < order[1] else a.T

    u = np.empty((len(slices), jj.size, ii.size), np.float32)
    v = np.empty_like(u)
    for k, (_, d, t) in enumerate(slices):
        u[k], v[k] = read(d, U_NAMES, "u", t), read(d, V_NAMES, "v", t)
        print("\r  slice %d/%d" % (k + 1, len(slices)), end="", file=sys.stderr)
    print(file=sys.stderr)

    dates = [s[0] for s in slices]
    seconds = np.array([(d - dates[0]).total_seconds() for d in dates])
    lat_out, lon_out = lat[jsl], lon[isl]
    if lat_out[0] > lat_out[-1]:
        lat_out, u, v = lat_out[::-1], u[:, ::-1, :], v[:, ::-1, :]

    with Dataset(args.out, "w", format="NETCDF3_64BIT_OFFSET") as out:
        out.createDimension("time", len(slices))
        out.createDimension("lat", lat_out.size)
        out.createDimension("lon", lon_out.size)
        tv = out.createVariable("time", "f8", ("time",))
        tv.units = "seconds since %s" % dates[0].strftime("%Y-%m-%d %H:%M:%S")
        tv[:] = seconds
        out.createVariable("lat", "f8", ("lat",))[:] = lat_out
        out.createVariable("lon", "f8", ("lon",))[:] = lon_out
        for name, arr in (("u", u), ("v", v)):
            var = out.createVariable(name, "f4", ("time", "lat", "lon"), fill_value=np.float32(np.nan))
            var.units = "m s-1"
            var[:] = arr
        out.source = " ".join(args.inputs)[:400]
        out.region = "%g,%g,%g,%g" % (lon_out[0], lon_out[-1], lat_out[0], lat_out[-1])

    land = np.isnan(u).any(axis=0).mean()
    print("wrote %s: %d slices (%s to %s), grid %d x %d at %.4g deg, %.1f%% land, %.1f MB" % (
        args.out, len(slices), dates[0], dates[-1], lon_out.size, lat_out.size,
        lon_out[1] - lon_out[0], 100 * land, (u.nbytes + v.nbytes) / 1e6))


if __name__ == "__main__":
    main()
