// Current fields and the velocity samplers the advection kernel calls (Step 3).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mpd {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kEarthRadius = 6371000.0;                       // m
constexpr double kMetresToDeg = 180.0 / (kPi * kEarthRadius);     // deg of latitude per metre

enum ParticleState : uint8_t { ALIVE = 0, BEACHED = 1, OUT = 2 };

// Regular grid of (u, v) per time slice, stored as contiguous arrays [t][j][i].
// Land nodes carry zero velocity and are flagged in `land`.
struct GridField {
    std::string name;
    int nx = 0, ny = 0, nt = 0;
    double x0 = 0, y0 = 0, dx = 1, dy = 1;
    bool periodic_x = false;  // global longitude grid
    bool spherical = false;   // x, y in degrees and u, v in m/s; otherwise Cartesian
    std::vector<double> times;  // seconds (spherical) or model time units, times[0] == 0
    std::vector<float> u, v;
    std::vector<uint8_t> land;  // [j][i]

    size_t slice_size() const { return (size_t)nx * ny; }
    const float* uslice(int k) const { return u.data() + (size_t)k * slice_size(); }
    const float* vslice(int k) const { return v.data() + (size_t)k * slice_size(); }
    double xmax() const { return x0 + (nx - 1) * dx; }
    double ymax() const { return y0 + (ny - 1) * dy; }
};

// Loads a classic-format NetCDF current file through the custom loader.
GridField load_netcdf_field(const std::string& path);

// Time-dependent double gyre on [0,2] x [0,1] (Shadden et al. 2005).
struct DoubleGyre {
    double eps = 0.15;
    double A = 0.1;
    double omega = 2.0 * kPi / 10.0;

    // tau is model time.
    inline void vel(double x, double y, double tau, double& dxdt, double& dydt) const {
        const double s = eps * std::sin(omega * tau);
        const double a = s, b = 1.0 - 2.0 * s;
        const double f = a * x * x + b * x;
        const double dfdx = 2.0 * a * x + b;
        dxdt = -kPi * A * std::sin(kPi * f) * std::cos(kPi * y);
        dydt = kPi * A * std::cos(kPi * f) * std::sin(kPi * y) * dfdx;
    }
    inline uint8_t classify(double& x, double& y) const {
        return (x < 0.0 || x > 2.0 || y < 0.0 || y > 1.0) ? OUT : ALIVE;
    }
};

// Samples the analytic double gyre onto a grid so it runs through the same
// interpolation kernel as the real data.
GridField make_double_gyre_grid(double eps, int nx, int ny, double slice_dt, double t_end);

// View of the two time slices bracketing the current step. tau in [0,1] is the
// position between them, so Step 3 is bilinear in space + linear in tau.
struct GridSampler {
    const float *u0, *v0, *u1, *v1;
    const uint8_t* land;
    int nx, ny;
    double x0, y0, inv_dx, inv_dy, xmax, ymax;
    bool periodic_x, spherical;

    GridSampler(const GridField& f, int k)
        : u0(f.uslice(k)), v0(f.vslice(k)), u1(f.uslice(k + 1)), v1(f.vslice(k + 1)),
          land(f.land.data()), nx(f.nx), ny(f.ny), x0(f.x0), y0(f.y0),
          inv_dx(1.0 / f.dx), inv_dy(1.0 / f.dy), xmax(f.xmax()), ymax(f.ymax()),
          periodic_x(f.periodic_x), spherical(f.spherical) {}

    inline void vel(double x, double y, double tau, double& dxdt, double& dydt) const {
        double fx = (x - x0) * inv_dx;
        double fy = (y - y0) * inv_dy;
        // RK4 stage points can step just outside the grid; clamp them to the edge.
        fy = fy < 0.0 ? 0.0 : (fy > ny - 1 ? ny - 1 : fy);
        int j = (int)fy;
        if (j > ny - 2) j = ny - 2;
        const double ty = fy - j;

        int i, i1;
        if (periodic_x) {
            fx -= nx * std::floor(fx / nx);
            i = (int)fx;
            if (i > nx - 1) i = nx - 1;
            i1 = (i + 1 == nx) ? 0 : i + 1;
        } else {
            fx = fx < 0.0 ? 0.0 : (fx > nx - 1 ? nx - 1 : fx);
            i = (int)fx;
            if (i > nx - 2) i = nx - 2;
            i1 = i + 1;
        }
        const double tx = fx - i;

        const size_t a00 = (size_t)j * nx + i, a10 = (size_t)j * nx + i1;
        const size_t a01 = a00 + nx, a11 = a10 + nx;
        const double w00 = (1.0 - tx) * (1.0 - ty), w10 = tx * (1.0 - ty);
        const double w01 = (1.0 - tx) * ty, w11 = tx * ty;

        const double ua = w00 * u0[a00] + w10 * u0[a10] + w01 * u0[a01] + w11 * u0[a11];
        const double ub = w00 * u1[a00] + w10 * u1[a10] + w01 * u1[a01] + w11 * u1[a11];
        const double va = w00 * v0[a00] + w10 * v0[a10] + w01 * v0[a01] + w11 * v0[a11];
        const double vb = w00 * v1[a00] + w10 * v1[a10] + w01 * v1[a01] + w11 * v1[a11];
        const double u = ua + tau * (ub - ua);
        const double v = va + tau * (vb - va);

        if (spherical) {
            double c = std::cos(y * kDegToRad);
            if (c < 1e-6) c = 1e-6;
            dxdt = u * kMetresToDeg / c;
            dydt = v * kMetresToDeg;
        } else {
            dxdt = u;
            dydt = v;
        }
    }

    // Step 5: where did the particle end up? Wraps x on a periodic grid.
    inline uint8_t classify(double& x, double& y) const {
        if (periodic_x) {
            x -= (nx / inv_dx) * std::floor((x - x0) * inv_dx / nx);
            if (!(y >= y0 && y <= ymax)) return OUT;
        } else if (!(x >= x0 && x <= xmax && y >= y0 && y <= ymax)) {
            return OUT;
        }
        int i = (int)((x - x0) * inv_dx + 0.5);
        int j = (int)((y - y0) * inv_dy + 0.5);
        if (i > nx - 1) i = periodic_x ? 0 : nx - 1;
        if (j > ny - 1) j = ny - 1;
        return land[(size_t)j * nx + i] ? BEACHED : ALIVE;
    }
};

}  // namespace mpd
