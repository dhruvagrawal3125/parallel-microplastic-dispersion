// Correctness tests for the loader, the interpolation, the integrator, the
// boundary rule and serial/parallel agreement.  Usage: mpd_tests TESTDATA_DIR
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mpd/advect.hpp"
#include "mpd/field.hpp"
#include "mpd/netcdf.hpp"
#include "mpd/particles.hpp"

using namespace mpd;

static int g_failed = 0, g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                    \
    do {                                                                         \
        ++g_checks;                                                              \
        const double a_ = (a), b_ = (b);                                         \
        if (!(std::fabs(a_ - b_) <= (tol))) {                                    \
            ++g_failed;                                                          \
            std::printf("  FAIL %s:%d  %s = %.12g, expected %.12g\n", __FILE__, __LINE__, #a, a_, b_); \
        }                                                                        \
    } while (0)

// Solid-body rotation about the origin; trajectories are exact circles.
struct Rotation {
    void vel(double x, double y, double, double& dx, double& dy) const { dx = -y; dy = x; }
    uint8_t classify(double&, double&) const { return ALIVE; }
};
struct RotationModel {
    using Sampler = Rotation;
    double dt;
    Sampler sampler(long step, double& tau, double& dtau) const { tau = step * dt; dtau = dt; return {}; }
};

// Small hand-built Cartesian field with u = au + bu*x + cu*y + du*t (same for v).
static GridField linear_field(int nx, int ny, int nt) {
    GridField f;
    f.nx = nx; f.ny = ny; f.nt = nt;
    f.x0 = 1.0; f.y0 = -2.0; f.dx = 0.5; f.dy = 0.25;
    f.land.assign(f.slice_size(), 0);
    for (int k = 0; k < nt; ++k) {
        f.times.push_back(10.0 * k);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = f.x0 + i * f.dx, y = f.y0 + j * f.dy, t = f.times[k];
                f.u.push_back((float)(0.5 + 0.25 * x - 0.125 * y + 0.0625 * t));
                f.v.push_back((float)(-1.0 + 0.5 * x + 0.75 * y - 0.03125 * t));
            }
    }
    return f;
}

static void test_interpolation() {
    std::printf("interpolation\n");
    const GridField f = linear_field(9, 7, 3);
    GridSampler s(f, 1);  // slices at t = 10 and 20
    const double pts[][3] = {{1.0, -2.0, 0.0}, {2.37, -1.11, 0.3}, {4.99, -0.51, 1.0}, {3.25, -1.5, 0.5}};
    for (const auto& p : pts) {
        double u, v;
        s.vel(p[0], p[1], p[2], u, v);
        const double t = 10.0 + 10.0 * p[2];
        CHECK_NEAR(u, 0.5 + 0.25 * p[0] - 0.125 * p[1] + 0.0625 * t, 1e-6);
        CHECK_NEAR(v, -1.0 + 0.5 * p[0] + 0.75 * p[1] - 0.03125 * t, 1e-6);
    }
    // Stage points outside the grid are clamped to the edge value, not extrapolated.
    double u_in, v_in, u_out, v_out;
    s.vel(f.xmax(), -1.0, 0.0, u_in, v_in);
    s.vel(f.xmax() + 3.0, -1.0, 0.0, u_out, v_out);
    CHECK_NEAR(u_out, u_in, 1e-12);

    // GridModel picks the right bracket and fraction.
    GridModel m{&f, 2.5};
    double tau, dtau;
    m.sampler(5, tau, dtau);  // t = 12.5 -> bracket [10, 20]
    CHECK_NEAR(tau, 0.25, 1e-12);
    CHECK_NEAR(dtau, 0.25, 1e-12);
    const GridSampler s0 = m.sampler(0, tau, dtau);
    CHECK(s0.u0 == f.uslice(0) && s0.u1 == f.uslice(1));
    const GridSampler s8 = m.sampler(8, tau, dtau);  // t = 20 -> clamped to the last bracket
    CHECK(s8.u1 == f.uslice(2));
}

static void test_rk4_order() {
    std::printf("RK4 convergence\n");
    double err[2];
    for (int r = 0; r < 2; ++r) {
        const long n = 40 << r;
        ParticlesSoA p;
        p.init({1.0}, {0.0});
        std::vector<double> busy(kBusyStride, 0.0);
        simulate(p, RotationModel{2.0 / n}, n, false, busy);
        err[r] = std::hypot(p.x(0) - std::cos(2.0), p.y(0) - std::sin(2.0));
    }
    std::printf("  error %.3g -> %.3g (ratio %.1f, 16 expected)\n", err[0], err[1], err[0] / err[1]);
    CHECK(err[0] < 1e-6);
    CHECK(err[0] / err[1] > 14.0 && err[0] / err[1] < 18.0);
}

static double psi(double x, double y) { return 0.1 * std::sin(kPi * x) * std::sin(kPi * y); }

static void test_double_gyre() {
    std::printf("double gyre\n");
    std::vector<double> rx, ry;
    seed_release(DoubleGyre{}, ReleaseBox{0.05, 1.95, 0.05, 0.95}, 400, true, 1, rx, ry);
    CHECK(rx.size() == 400);
    std::vector<double> busy(kBusyStride, 0.0);

    // eps = 0 is steady, so the streamfunction is conserved along every path.
    ParticlesSoA p;
    p.init(rx, ry);
    simulate(p, AnalyticModel{DoubleGyre{0.0}, 0.05}, 400, false, busy);
    double worst = 0;
    for (size_t i = 0; i < p.size(); ++i)
        worst = std::fmax(worst, std::fabs(psi(p.x(i), p.y(i)) - psi(rx[i], ry[i])));
    std::printf("  max streamfunction change at eps=0: %.3g\n", worst);
    CHECK(worst < 1e-7);

    // Gridded field vs the closed-form field integrated with a much finer step:
    // the error must shrink as the grid and slice spacing are refined.
    const double eps = 0.15, T = 5.0, dt = 0.05;
    const long n = std::lround(T / dt);
    ParticlesSoA ref;
    ref.init(rx, ry);
    simulate(ref, AnalyticModel{DoubleGyre{eps}, dt / 16}, n * 16, false, busy);
    double err[2];
    for (int r = 0; r < 2; ++r) {
        const int nx = r ? 401 : 101;
        const GridField f = make_double_gyre_grid(eps, nx, (nx + 1) / 2, r ? 0.05 : 0.2, T);
        ParticlesSoA q;
        q.init(rx, ry);
        simulate(q, GridModel{&f, dt}, n, false, busy);
        double sum = 0;
        for (size_t i = 0; i < q.size(); ++i) sum += std::hypot(q.x(i) - ref.x(i), q.y(i) - ref.y(i));
        err[r] = sum / (double)q.size();
    }
    std::printf("  mean error vs analytic: %.3g (101x51) -> %.3g (401x201)\n", err[0], err[1]);
    CHECK(err[1] < 1e-4);
    CHECK(err[0] / err[1] > 8.0);
}

template <class P> static void snapshot(const P& p, std::vector<double>& xy, std::vector<uint8_t>& st) {
    xy.clear(); st.clear();
    for (size_t i = 0; i < p.size(); ++i) { xy.push_back(p.x(i)); xy.push_back(p.y(i)); st.push_back(p.state(i)); }
}

static void test_serial_parallel_agree(const std::string& dir) {
    std::printf("serial vs parallel\n");
    const GridField f = load_netcdf_field(dir + "/demo_eddies.nc");
    const GridModel m{&f, 3600.0};
    const long nsteps = 24 * 20;
    std::vector<double> rx, ry;
    double tau, dtau;
    seed_release(m.sampler(0, tau, dtau), ReleaseBox{f.x0, f.xmax(), f.y0, f.ymax()}, 20000, true, 1, rx, ry);

    std::vector<double> busy(64 * kBusyStride, 0.0), ref_xy, xy;
    std::vector<uint8_t> ref_st, st;
    ParticlesSoA base;
    base.init(rx, ry);
    simulate(base, m, nsteps, false, busy);
    snapshot(base, ref_xy, ref_st);
    size_t beached = 0, out = 0;
    for (uint8_t s : ref_st) { beached += (s == BEACHED); out += (s == OUT); }
    std::printf("  %zu beached, %zu left the domain of %zu\n", beached, out, ref_st.size());
    CHECK(beached > 0);

    ParticlesAoS aos;
    aos.init(rx, ry);
    simulate(aos, m, nsteps, false, busy);
    snapshot(aos, xy, st);
    CHECK(xy == ref_xy && st == ref_st);

#ifdef _OPENMP
    const omp_sched_t kinds[] = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
    for (omp_sched_t kind : kinds)
        for (int threads : {2, 4}) {
            omp_set_num_threads(threads);
            omp_set_schedule(kind, kind == omp_sched_dynamic ? 256 : 0);
            ParticlesSoA a;
            a.init(rx, ry);
            simulate(a, m, nsteps, true, busy);
            snapshot(a, xy, st);
            CHECK(xy == ref_xy && st == ref_st);
            ParticlesAoS b;
            b.init(rx, ry);
            simulate(b, m, nsteps, true, busy);
            snapshot(b, xy, st);
            CHECK(xy == ref_xy && st == ref_st);
        }
#else
    std::printf("  (built without OpenMP: parallel schedules not exercised)\n");
    ParticlesSoA a;
    a.init(rx, ry);
    simulate(a, m, nsteps, true, busy);
    snapshot(a, xy, st);
    CHECK(xy == ref_xy && st == ref_st);
#endif
}

static void test_boundary_rule() {
    std::printf("boundary rule\n");
    // Uniform eastward flow of 1 unit/time on a 21x5 grid with a land wall at i >= 10.
    GridField f;
    f.nx = 21; f.ny = 5; f.nt = 2;
    f.dx = f.dy = 1.0;
    f.times = {0.0, 100.0};
    f.u.assign(2 * f.slice_size(), 1.0f);
    f.v.assign(2 * f.slice_size(), 0.0f);
    f.land.assign(f.slice_size(), 0);
    for (int j = 0; j < f.ny; ++j)
        for (int i = 10; i < f.nx; ++i) {
            f.land[j * f.nx + i] = 1;
            f.u[j * f.nx + i] = f.u[f.slice_size() + j * f.nx + i] = 0.0f;
        }
    std::vector<double> busy(kBusyStride, 0.0);
    ParticlesSoA p;
    p.init({2.0, 2.0}, {2.0, 2.0});
    simulate(p, GridModel{&f, 0.25}, 200, false, busy);
    CHECK(p.state(0) == BEACHED);
    CHECK(p.x(0) < 9.5 && p.x(0) > 8.5);  // frozen at the last wet position, short of the wall
    CHECK(p.ages[0] < 200);               // and no longer advanced

    // Westward flow carries a particle out of a non-periodic domain.
    for (auto& u : f.u) u = -1.0f;
    ParticlesSoA q;
    q.init({2.0}, {2.0});
    simulate(q, GridModel{&f, 0.25}, 200, false, busy);
    CHECK(q.state(0) == OUT);
    CHECK(q.x(0) >= 0.0 && q.x(0) < 0.3);

    // On a periodic (global) grid the same particle wraps around instead.
    GridField g = f;
    g.periodic_x = true;
    g.land.assign(g.slice_size(), 0);
    ParticlesSoA w;
    w.init({2.0}, {2.0});
    simulate(w, GridModel{&g, 0.25}, 40, false, busy);  // moves 10 units west: 2 -> -8 -> 13
    CHECK(w.state(0) == ALIVE);
    CHECK_NEAR(w.x(0), 13.0, 1e-9);
}

static void test_netcdf(const std::string& dir) {
    std::printf("NetCDF loader\n");
    for (const char* name : {"linear_cdf1.nc", "linear_cdf2.nc", "linear_cdf5.nc", "packed.nc"}) {
        const bool packed = std::string(name) == "packed.nc";
        const double tol = packed ? 6e-4 : 2e-6;  // packed values are rounded to 1 mm/s
        NcFile nc(dir + "/" + name);
        CHECK(nc.var("lat") && nc.var("lon") && nc.var("time"));
        const GridField f = load_netcdf_field(dir + "/" + name);
        CHECK(f.nx == 21 && f.ny == 41 && f.nt == 4);
        CHECK(f.spherical && !f.periodic_x);
        CHECK_NEAR(f.x0, 10.0, 1e-6);
        CHECK_NEAR(f.dx, 0.5, 1e-6);
        CHECK_NEAR(f.y0, -5.0, 1e-6);
        CHECK_NEAR(f.dy, 0.25, 1e-6);
        CHECK_NEAR(f.times[1], 86400.0, 1e-6);
        CHECK_NEAR(f.times[3], 3 * 86400.0, 1e-6);
        size_t nland = 0;
        double worst = 0;
        for (int k = 0; k < f.nt; ++k)
            for (int j = 0; j < f.ny; ++j)
                for (int i = 0; i < f.nx; ++i) {
                    const size_t o = (size_t)j * f.nx + i;
                    const bool island = (j >= 18 && j < 22 && i >= 8 && i < 12);
                    if (k == 0) nland += f.land[o];
                    if (island) {
                        CHECK(f.land[o] == 1 && f.uslice(k)[o] == 0.0f);
                        continue;
                    }
                    const double lon = 10.0 + 0.5 * i, lat = -5.0 + 0.25 * j, t = f.times[k];
                    worst = std::fmax(worst, std::fabs(f.uslice(k)[o] - (0.01 * lon + 0.02 * lat + 1e-6 * t)));
                    worst = std::fmax(worst, std::fabs(f.vslice(k)[o] - (-0.03 * lon + 0.005 * lat - 2e-6 * t)));
                }
        std::printf("  %-16s CDF-%d  max value error %.2g\n", name, nc.version(), worst);
        CHECK(nland == 16);
        CHECK(worst < tol);
    }
    // Velocity conversion on the sphere: 1 m/s north for one day is 86.4 km.
    GridField f = load_netcdf_field(dir + "/linear_cdf2.nc");
    std::fill(f.u.begin(), f.u.end(), 0.0f);
    std::fill(f.v.begin(), f.v.end(), 1.0f);
    std::fill(f.land.begin(), f.land.end(), 0);
    ParticlesSoA p;
    p.init({15.0}, {0.0});
    std::vector<double> busy(kBusyStride, 0.0);
    simulate(p, GridModel{&f, 3600.0}, 24, false, busy);
    CHECK_NEAR(p.y(0), 86400.0 / kEarthRadius / kDegToRad, 1e-9);
    CHECK_NEAR(p.x(0), 15.0, 1e-12);
}

static void test_release() {
    std::printf("release points\n");
    const DoubleGyre g;
    for (bool lattice : {true, false}) {
        std::vector<double> ax, ay, bx, by;
        seed_release(g, ReleaseBox{0.2, 1.8, 0.1, 0.9}, 12345, lattice, 42, ax, ay);
        seed_release(g, ReleaseBox{0.2, 1.8, 0.1, 0.9}, 12345, lattice, 42, bx, by);
        CHECK(ax.size() == 12345 && ax == bx && ay == by);
        bool inside = true;
        for (size_t i = 0; i < ax.size(); ++i)
            inside = inside && ax[i] >= 0.2 && ax[i] <= 1.8 && ay[i] >= 0.1 && ay[i] <= 0.9;
        CHECK(inside);
    }
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "build/testdata";
    test_interpolation();
    test_rk4_order();
    test_double_gyre();
    test_boundary_rule();
    test_release();
    test_netcdf(dir);
    test_serial_parallel_agree(dir);
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
