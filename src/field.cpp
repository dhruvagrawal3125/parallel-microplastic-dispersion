#include "mpd/field.hpp"

#include <algorithm>
#include <stdexcept>

#include "mpd/netcdf.hpp"

namespace mpd {

namespace {

enum Axis { AX_T, AX_Y, AX_X, AX_OTHER };

bool contains(const std::vector<std::string>& names, const std::string& n) {
    return std::find(names.begin(), names.end(), n) != names.end();
}

const std::vector<std::string> kTimeNames = {"time", "t", "MT"};
const std::vector<std::string> kLatNames = {"lat", "latitude", "Latitude", "y"};
const std::vector<std::string> kLonNames = {"lon", "longitude", "Longitude", "x"};
const std::vector<std::string> kUNames = {"u", "water_u", "uo", "ugos", "U"};
const std::vector<std::string> kVNames = {"v", "water_v", "vo", "vgos", "V"};

double att_num(const NcVar& v, const char* name, double fallback) {
    const NcAttr* a = v.att(name);
    return (a && !a->num.empty()) ? a->num[0] : fallback;
}

// Seconds per unit of a CF time axis ("hours since 2000-01-01 00:00:00").
double time_unit_seconds(const NcVar& t) {
    const NcAttr* a = t.att("units");
    if (!a) return 1.0;
    const std::string& s = a->text;
    if (s.rfind("sec", 0) == 0) return 1.0;
    if (s.rfind("min", 0) == 0) return 60.0;
    if (s.rfind("hour", 0) == 0) return 3600.0;
    if (s.rfind("day", 0) == 0) return 86400.0;
    throw std::runtime_error("field: unsupported time units '" + s + "'");
}

// Checks that a coordinate axis is evenly spaced and returns (first, spacing).
void regular_axis(const std::vector<double>& c, const char* what, double& first, double& step) {
    if (c.size() < 2) throw std::runtime_error(std::string("field: need at least 2 points along ") + what);
    first = c.front();
    step = (c.back() - c.front()) / (double)(c.size() - 1);
    for (size_t i = 0; i < c.size(); ++i)
        if (std::fabs(c[i] - (first + step * (double)i)) > 1e-2 * std::fabs(step))
            throw std::runtime_error(std::string("field: ") + what + " axis is not evenly spaced");
}

// Reads one velocity component into [t][j][i] order with land set to NaN.
void read_component(const NcFile& nc, const NcVar& var, const Axis axes[], int nt, int ny, int nx,
                    bool flip_y, std::vector<float>& out) {
    std::vector<float> raw;
    nc.read(var, raw);

    const double scale = att_num(var, "scale_factor", 1.0);
    const double offset = att_num(var, "add_offset", 0.0);
    const double fill = att_num(var, "_FillValue", NAN);
    const double missing = att_num(var, "missing_value", NAN);

    // Stride of each logical axis in the raw array.
    size_t stride[3] = {0, 0, 0}, s = 1;
    for (int d = (int)var.shape.size() - 1; d >= 0; --d) {
        if (axes[d] != AX_OTHER) stride[axes[d]] = s;
        s *= var.shape[d];
    }

    out.resize((size_t)nt * ny * nx);
    size_t o = 0;
    for (int t = 0; t < nt; ++t)
        for (int j = 0; j < ny; ++j) {
            const int jj = flip_y ? ny - 1 - j : j;
            const size_t base = t * stride[AX_T] + jj * stride[AX_Y];
            for (int i = 0; i < nx; ++i, ++o) {
                const float r = raw[base + i * stride[AX_X]];
                const bool bad = std::isnan(r) || r == (float)fill || r == (float)missing;
                const double val = bad ? NAN : r * scale + offset;
                out[o] = (std::isnan(val) || std::fabs(val) > 1e3) ? NAN : (float)val;
            }
        }
}

}  // namespace

GridField load_netcdf_field(const std::string& path) {
    NcFile nc(path);
    const NcVar* uvar = nc.find_var(kUNames);
    const NcVar* vvar = nc.find_var(kVNames);
    const NcVar* latv = nc.find_var(kLatNames);
    const NcVar* lonv = nc.find_var(kLonNames);
    const NcVar* timev = nc.find_var(kTimeNames);
    if (!uvar || !vvar) throw std::runtime_error("field: no u/v velocity variables in " + path);
    if (!latv || !lonv) throw std::runtime_error("field: no lat/lon coordinate variables in " + path);
    if (uvar->shape != vvar->shape) throw std::runtime_error("field: u and v have different shapes");

    // Map each dimension of u onto time / lat / lon; anything else must be length 1.
    std::vector<Axis> axes(uvar->dimids.size(), AX_OTHER);
    int nt = 1, ny = 0, nx = 0;
    for (size_t d = 0; d < axes.size(); ++d) {
        const std::string& dn = nc.dims()[uvar->dimids[d]].name;
        const int len = (int)uvar->shape[d];
        if (contains(kTimeNames, dn)) { axes[d] = AX_T; nt = len; }
        else if (contains(kLatNames, dn)) { axes[d] = AX_Y; ny = len; }
        else if (contains(kLonNames, dn)) { axes[d] = AX_X; nx = len; }
        else if (len != 1)
            throw std::runtime_error("field: unexpected dimension '" + dn + "' of length " +
                                     std::to_string(len) + " (subset to one depth level first)");
    }
    if (nx < 2 || ny < 2) throw std::runtime_error("field: u must have lat and lon dimensions");

    std::vector<double> lat, lon, time;
    nc.read(*latv, lat);
    nc.read(*lonv, lon);
    if ((int)lat.size() != ny || (int)lon.size() != nx)
        throw std::runtime_error("field: coordinate lengths do not match u");
    const bool flip_y = lat.back() < lat.front();
    if (flip_y) std::reverse(lat.begin(), lat.end());

    GridField f;
    f.name = path;
    f.nx = nx;
    f.ny = ny;
    f.spherical = true;
    regular_axis(lon, "longitude", f.x0, f.dx);
    regular_axis(lat, "latitude", f.y0, f.dy);
    if (f.dx <= 0) throw std::runtime_error("field: longitude must be increasing");
    f.periodic_x = std::fabs(nx * f.dx - 360.0) < 0.5 * f.dx;

    read_component(nc, *uvar, axes.data(), nt, ny, nx, flip_y, f.u);
    read_component(nc, *vvar, axes.data(), nt, ny, nx, flip_y, f.v);

    if (timev && nt > 1) {
        nc.read(*timev, time);
        const double unit = time_unit_seconds(*timev);
        for (int k = 0; k < nt; ++k) {
            f.times.push_back((time[k] - time[0]) * unit);
            if (k > 0 && f.times[k] <= f.times[k - 1])
                throw std::runtime_error("field: time axis must be strictly increasing");
        }
    } else {
        // A single snapshot is treated as a steady field: duplicate the slice.
        nt = 2;
        f.times = {0.0, 365.0 * 86400.0};
        f.u.insert(f.u.end(), f.u.begin(), f.u.begin() + (std::ptrdiff_t)f.slice_size());
        f.v.insert(f.v.end(), f.v.begin(), f.v.begin() + (std::ptrdiff_t)f.slice_size());
    }
    f.nt = nt;

    // A node is land if it is missing in any slice; land carries zero velocity.
    const size_t n = f.slice_size();
    f.land.assign(n, 0);
    for (int k = 0; k < nt; ++k)
        for (size_t p = 0; p < n; ++p)
            if (std::isnan(f.u[k * n + p]) || std::isnan(f.v[k * n + p])) f.land[p] = 1;
    for (int k = 0; k < nt; ++k)
        for (size_t p = 0; p < n; ++p)
            if (f.land[p]) f.u[k * n + p] = f.v[k * n + p] = 0.0f;
    return f;
}

GridField make_double_gyre_grid(double eps, int nx, int ny, double slice_dt, double t_end) {
    if (nx < 2 || ny < 2 || slice_dt <= 0) throw std::runtime_error("double gyre: bad grid parameters");
    GridField f;
    f.name = "double_gyre";
    f.nx = nx;
    f.ny = ny;
    f.x0 = 0.0;
    f.y0 = 0.0;
    f.dx = 2.0 / (nx - 1);
    f.dy = 1.0 / (ny - 1);
    f.nt = std::max(2, (int)std::ceil(t_end / slice_dt - 1e-9) + 1);
    f.land.assign(f.slice_size(), 0);
    f.u.resize((size_t)f.nt * f.slice_size());
    f.v.resize(f.u.size());

    DoubleGyre g;
    g.eps = eps;
    size_t o = 0;
    for (int k = 0; k < f.nt; ++k) {
        f.times.push_back(k * slice_dt);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i, ++o) {
                double u, v;
                g.vel(i * f.dx, j * f.dy, f.times[k], u, v);
                f.u[o] = (float)u;
                f.v[o] = (float)v;
            }
    }
    return f;
}

}  // namespace mpd
