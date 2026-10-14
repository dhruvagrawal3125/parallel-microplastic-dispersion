// The advection kernel (Steps 3-5) and the serial / OpenMP timestep loops.
// Both loops call the same rk4_step, so the serial baseline and the parallel
// version share their numerics exactly.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mpd/field.hpp"

namespace mpd {

// One fixed-step RK4 update of a single particle, followed by the boundary rule:
// a particle whose new position is on land or outside the domain is frozen at its
// last valid position. tau/dtau are the sampler's time coordinate for this step.
template <class Sampler>
inline void rk4_step(const Sampler& s, double& x, double& y, uint8_t& state, double tau, double dtau,
                     double dt) {
    double k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y;
    const double h = 0.5 * dt;
    s.vel(x, y, tau, k1x, k1y);
    s.vel(x + h * k1x, y + h * k1y, tau + 0.5 * dtau, k2x, k2y);
    s.vel(x + h * k2x, y + h * k2y, tau + 0.5 * dtau, k3x, k3y);
    s.vel(x + dt * k3x, y + dt * k3y, tau + dtau, k4x, k4y);
    double nx = x + (dt / 6.0) * (k1x + 2.0 * (k2x + k3x) + k4x);
    double ny = y + (dt / 6.0) * (k1y + 2.0 * (k2y + k3y) + k4y);
    const uint8_t st = s.classify(nx, ny);
    if (st == ALIVE) {
        x = nx;
        y = ny;
    } else {
        state = st;
    }
}

template <class P, class Sampler>
void step_serial(P& p, const Sampler& sampler, double tau, double dtau, double dt) {
    const Sampler s = sampler;
    const std::ptrdiff_t n = (std::ptrdiff_t)p.size();
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        if (p.state(i) != ALIVE) continue;
        rk4_step(s, p.x(i), p.y(i), p.state(i), tau, dtau, dt);
        ++p.age(i);
    }
}

// Each thread's slot in the busy-time array is padded to its own cache line.
constexpr int kBusyStride = 8;

// Algorithm 1. The schedule comes from omp_set_schedule (schedule(runtime)), so
// one binary covers static, dynamic and guided. `busy` accumulates, per thread,
// the time from entering the loop to finishing its own share of it (nowait).
template <class P, class Sampler>
void step_parallel(P& p, const Sampler& sampler, double tau, double dtau, double dt, double* busy) {
    const std::ptrdiff_t n = (std::ptrdiff_t)p.size();
#ifdef _OPENMP
    #pragma omp parallel
    {
        const Sampler s = sampler;
        const double t0 = omp_get_wtime();
        #pragma omp for schedule(runtime) nowait
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            if (p.state(i) != ALIVE) continue;
            rk4_step(s, p.x(i), p.y(i), p.state(i), tau, dtau, dt);
            ++p.age(i);
        }
        busy[omp_get_thread_num() * kBusyStride] += omp_get_wtime() - t0;
    }
#else
    const auto t0 = std::chrono::steady_clock::now();
    step_serial(p, sampler, tau, dtau, dt);
    busy[0] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    (void)n;
#endif
}

// Step 1 for gridded data: picks the two slices bracketing each step.
struct GridModel {
    using Sampler = GridSampler;
    const GridField* field;
    double dt;

    Sampler sampler(long step, double& tau, double& dtau) const {
        const std::vector<double>& ts = field->times;
        const double t = step * dt;
        int k = (int)(std::upper_bound(ts.begin(), ts.end(), t) - ts.begin()) - 1;
        k = std::max(0, std::min(k, field->nt - 2));
        const double span = ts[k + 1] - ts[k];
        tau = (t - ts[k]) / span;
        dtau = dt / span;
        return GridSampler(*field, k);
    }
};

// The double gyre evaluated in closed form at every stage (no grid, no interpolation).
struct AnalyticModel {
    using Sampler = DoubleGyre;
    DoubleGyre gyre;
    double dt;

    Sampler sampler(long step, double& tau, double& dtau) const {
        tau = step * dt;
        dtau = dt;
        return gyre;
    }
};

// Positions of every `stride`-th particle, sampled every `every` steps. The
// buffer is allocated up front so recording never touches the disk mid-run.
struct TrajRecorder {
    long every = 0;
    size_t stride = 1, tracked = 0;
    std::vector<float> xy;  // [snapshot][particle][2]

    void setup(size_t nparticles, size_t want, long every_steps, long nsteps) {
        every = every_steps;
        tracked = std::min(want, nparticles);
        stride = tracked ? nparticles / tracked : 1;
        xy.clear();
        xy.reserve(tracked * 2 * (size_t)(nsteps / std::max(1L, every) + 2));
    }
    template <class P> void record(const P& p) {
        for (size_t k = 0; k < tracked; ++k) {
            xy.push_back((float)p.x(k * stride));
            xy.push_back((float)p.y(k * stride));
        }
    }
    size_t snapshots() const { return tracked ? xy.size() / (2 * tracked) : 0; }
};

// Runs `nsteps` timesteps and returns the wall-clock seconds they took.
template <class P, class Model>
double simulate(P& p, const Model& m, long nsteps, bool parallel, std::vector<double>& busy,
                TrajRecorder* traj = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    if (traj && traj->every > 0) traj->record(p);
    for (long step = 0; step < nsteps; ++step) {
        double tau, dtau;
        const typename Model::Sampler s = m.sampler(step, tau, dtau);
        if (parallel)
            step_parallel(p, s, tau, dtau, m.dt, busy.data());
        else
            step_serial(p, s, tau, dtau, m.dt);
        if (traj && traj->every > 0 && (step + 1) % traj->every == 0) traj->record(p);
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace mpd
