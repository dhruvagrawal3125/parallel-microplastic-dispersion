#!/usr/bin/env python3
"""Writes small NetCDF files with known contents for the loader tests, plus a
synthetic eddy field with an island (demo_eddies.nc) for trying the real-data
path without downloading anything.

    python3 scripts/make_test_nc.py OUTDIR
"""
import os
import sys

import numpy as np
from netCDF4 import Dataset

LON = np.arange(10.0, 20.0 + 1e-9, 0.5)   # 21
LAT = np.arange(-5.0, 5.0 + 1e-9, 0.25)   # 41
HOURS = np.array([0.0, 24.0, 48.0, 72.0])
ISLAND = (slice(18, 22), slice(8, 12))    # [lat, lon] block marked as land


def linear_uv():
    """u, v linear in lon, lat and time so bilinear + time interpolation is exact."""
    t = (HOURS * 3600.0)[:, None, None]
    la = LAT[None, :, None]
    lo = LON[None, None, :]
    u = 0.01 * lo + 0.02 * la + 1e-6 * t
    v = -0.03 * lo + 0.005 * la - 2e-6 * t
    return u, v


def write_float(path, fmt):
    """(time, lat, lon) float32 with an unlimited time dimension and NaN land."""
    u, v = linear_uv()
    u[:, ISLAND[0], ISLAND[1]] = np.nan
    v[:, ISLAND[0], ISLAND[1]] = np.nan
    with Dataset(path, "w", format=fmt) as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", LAT.size)
        ds.createDimension("lon", LON.size)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "hours since 2020-01-01 00:00:00"
        ds.createVariable("lat", "f8", ("lat",))[:] = LAT
        ds.createVariable("lon", "f8", ("lon",))[:] = LON
        uv = ds.createVariable("u", "f4", ("time", "lat", "lon"), fill_value=np.float32(np.nan))
        vv = ds.createVariable("v", "f4", ("time", "lat", "lon"), fill_value=np.float32(np.nan))
        tv[:] = HOURS
        uv[:] = u
        vv[:] = v


def write_packed(path):
    """HYCOM/OSCAR-style quirks: packed int16, a depth axis, descending latitude,
    (time, depth, lon, lat) dimension order, and a missing-value sentinel."""
    u, v = linear_uv()
    fill = -30000
    scale = 0.001

    def pack(a):
        p = np.round(a / scale).astype(np.int16)
        p[:, ISLAND[0], ISLAND[1]] = fill
        return p[:, ::-1, :].transpose(0, 2, 1)[:, None, :, :]  # flip lat, then (t, depth, lon, lat)

    with Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.set_auto_maskandscale(False)
        ds.createDimension("time", HOURS.size)
        ds.createDimension("depth", 1)
        ds.createDimension("lon", LON.size)
        ds.createDimension("lat", LAT.size)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "days since 2020-01-01"
        tv[:] = HOURS / 24.0
        ds.createVariable("depth", "f4", ("depth",))[:] = [0.0]
        ds.createVariable("lon", "f4", ("lon",))[:] = LON
        ds.createVariable("lat", "f4", ("lat",))[:] = LAT[::-1]
        for name, arr in (("water_u", u), ("water_v", v)):
            var = ds.createVariable(name, "i2", ("time", "depth", "lon", "lat"))
            var.set_auto_maskandscale(False)
            var.scale_factor = np.float32(scale)
            var.add_offset = np.float32(0.0)
            var.missing_value = np.int16(fill)
            var[:] = pack(arr)


def write_demo(path):
    """A western-boundary jet plus drifting eddies around an island: enough
    structure and coastline to exercise the land-mask rule and make a map."""
    lon = np.arange(-70.0, -40.0 + 1e-9, 0.125)
    lat = np.arange(28.0, 44.0 + 1e-9, 0.125)
    days = np.arange(0, 61)
    lo, la = np.meshgrid(lon, lat)
    u = np.empty((days.size, lat.size, lon.size), np.float32)
    v = np.empty_like(u)
    rng = np.random.default_rng(7)
    centres = rng.uniform([-68, 30], [-42, 42], size=(14, 2))
    sign = rng.choice([-1.0, 1.0], size=14)
    for k, d in enumerate(days):
        # Meandering eastward jet, expressed through a streamfunction so the flow is non-divergent.
        axis = 36.0 + 1.5 * np.sin(np.radians(12.0 * (lo + 70.0)) - 0.1 * d)
        psi = -1.2 * np.tanh((la - axis) / 1.0)
        for (cx, cy), s in zip(centres, sign):
            cxd = cx - 0.05 * d
            psi += s * 0.6 * np.exp(-(((lo - cxd) ** 2 + (la - cy) ** 2) / 1.5 ** 2))
        dpsi_dy, dpsi_dx = np.gradient(psi, 0.125, 0.125)
        u[k] = -dpsi_dy * 0.35
        v[k] = dpsi_dx * 0.35
    land = ((lo + 64.7) ** 2 + (la - 32.3) ** 2 < 0.6 ** 2) | (la > 43.2 + 0.03 * (lo + 55.0))
    u[:, land] = np.nan
    v[:, land] = np.nan
    with Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("time", days.size)
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "days since 2024-01-01"
        tv[:] = days
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("u", "f4", ("time", "lat", "lon"), fill_value=np.float32(np.nan))[:] = u
        ds.createVariable("v", "f4", ("time", "lat", "lon"), fill_value=np.float32(np.nan))[:] = v
        ds.note = "Synthetic jet + eddies for testing; NOT real ocean data."


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "build/testdata"
    os.makedirs(out, exist_ok=True)
    write_float(os.path.join(out, "linear_cdf2.nc"), "NETCDF3_64BIT_OFFSET")
    write_float(os.path.join(out, "linear_cdf1.nc"), "NETCDF3_CLASSIC")
    write_float(os.path.join(out, "linear_cdf5.nc"), "NETCDF3_64BIT_DATA")
    write_packed(os.path.join(out, "packed.nc"))
    write_demo(os.path.join(out, "demo_eddies.nc"))
    print("wrote test NetCDF files to", out)


if __name__ == "__main__":
    main()
