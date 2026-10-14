// mpd: microplastic dispersion driver. Runs the serial baseline and/or the
// OpenMP advection kernel on a double-gyre or NetCDF current field, times it
// (warm-ups + timed runs, median reported) and writes results for plotting.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mpd/advect.hpp"
#include "mpd/field.hpp"
#include "mpd/npy.hpp"
#include "mpd/particles.hpp"

using namespace mpd;

namespace {

const char* kUsage = R"(usage: mpd [options]

Current field
  --field dg|nc           double gyre (default) or NetCDF file
  --file PATH             classic NetCDF current file (with --field nc)
  --eps E                 double-gyre perturbation amplitude        [0.15]
  --dg-grid NXxNY         double-gyre grid resolution               [401x201]
  --dg-slice-dt T         spacing of double-gyre time slices        [0.1]
  --analytic              evaluate the double gyre in closed form (no grid)

Simulation
  --particles P           number of particles                       [100000]
  --dt DT                 timestep: seconds (nc) or model time (dg) [3600 | 0.05]
  --duration D            run length: days (nc) or model time (dg)  [whole file | 20]
  --release lattice|random   release-point pattern                  [lattice]
  --release-box X0,X1,Y0,Y1  release region                         [whole domain]
  --seed S                seed for random release                   [42]
  --release-file FILE     release points from an (N,2) float64 .npy of x,y; sets P = N

Execution
  --mode serial|parallel|both                                       [both]
  --threads N             OpenMP threads                            [all]
  --schedule static|dynamic|guided                                  [static]
  --chunk C               schedule chunk size; 0 = static/guided default, 1024 for dynamic
  --layout soa|aos        particle memory layout                    [soa]
  --warmup W --runs R     untimed and timed repetitions             [2, 5]

Output
  --csv FILE              append one result row per mode
  --out PREFIX            write PREFIX_{final,state,release,traj,land}.npy and PREFIX_meta.json
  --traj N                particles to track for trajectories       [200]
  --traj-every K          steps between trajectory samples          [auto]
  --ref FILE              reference final positions (.npy) for the drift check
  --truth                 (dg) also report error against a fine-step analytic reference
)";

struct Cfg {
    std::string field = "dg", file, mode = "both", schedule = "static", layout = "soa";
    std::string release = "lattice", release_box, release_file, csv, out, ref;
    double eps = 0.15, dg_slice_dt = 0.1, dt = -1, duration = -1;
    int dg_nx = 401, dg_ny = 201, threads = 0, chunk = 0, warmup = 2, runs = 5;
    size_t particles = 100000, traj = 200;
    long traj_every = 0;
    uint64_t seed = 42;
    bool analytic = false, truth = false;
};

Cfg parse(int argc, char** argv) {
    Cfg c;
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { std::cout << kUsage; std::exit(0); }
        if (a.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument '" + a + "'");
        a = a.substr(2);
        if (a == "analytic" || a == "truth") { kv[a] = "1"; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value for --" + a);
        kv[a] = argv[++i];
    }
    auto take = [&kv](const char* k, std::string& dst) {
        auto it = kv.find(k);
        if (it == kv.end()) return false;
        dst = it->second;
        kv.erase(it);
        return true;
    };
    std::string s;
    take("field", c.field); take("file", c.file); take("mode", c.mode);
    take("schedule", c.schedule); take("layout", c.layout); take("release", c.release);
    take("release-box", c.release_box); take("release-file", c.release_file); take("csv", c.csv); take("out", c.out); take("ref", c.ref);
    if (take("eps", s)) c.eps = std::stod(s);
    if (take("dg-slice-dt", s)) c.dg_slice_dt = std::stod(s);
    if (take("dt", s)) c.dt = std::stod(s);
    if (take("duration", s)) c.duration = std::stod(s);
    if (take("threads", s)) c.threads = std::stoi(s);
    if (take("chunk", s)) c.chunk = std::stoi(s);
    if (take("warmup", s)) c.warmup = std::stoi(s);
    if (take("runs", s)) c.runs = std::stoi(s);
    if (take("particles", s)) c.particles = (size_t)std::stod(s);  // accepts 1e6
    if (take("traj", s)) c.traj = (size_t)std::stod(s);
    if (take("traj-every", s)) c.traj_every = std::stol(s);
    if (take("seed", s)) c.seed = std::stoull(s);
    if (take("analytic", s)) c.analytic = true;
    if (take("truth", s)) c.truth = true;
    if (take("dg-grid", s)) {
        const size_t x = s.find('x');
        if (x == std::string::npos) throw std::runtime_error("--dg-grid expects NXxNY");
        c.dg_nx = std::stoi(s.substr(0, x));
        c.dg_ny = std::stoi(s.substr(x + 1));
    }
    if (!kv.empty()) throw std::runtime_error("unknown option --" + kv.begin()->first);

    auto one_of = [](const std::string& v, std::initializer_list<const char*> ok, const char* what) {
        for (const char* o : ok)
            if (v == o) return;
        throw std::runtime_error(std::string("bad value '") + v + "' for --" + what);
    };
    one_of(c.field, {"dg", "nc"}, "field");
    one_of(c.mode, {"serial", "parallel", "both"}, "mode");
    one_of(c.schedule, {"static", "dynamic", "guided"}, "schedule");
    one_of(c.layout, {"soa", "aos"}, "layout");
    one_of(c.release, {"lattice", "random"}, "release");
    if (c.field == "nc" && c.file.empty()) throw std::runtime_error("--field nc needs --file");
    if (c.runs < 1 || c.warmup < 0) throw std::runtime_error("--runs must be >= 1 and --warmup >= 0");
    if (c.particles < 1) throw std::runtime_error("--particles must be >= 1");
    return c;
}

// What the run was on; shared by the CSV row and the metadata file.
struct Meta {
    std::string field_name;
    double eps = 0;
    int nx = 0, ny = 0, nt = 0;
    bool spherical = false;
    double x0 = 0, x1 = 0, y0 = 0, y1 = 0, dt = 0;
    long nsteps = 0;
    const GridField* grid = nullptr;
};

struct Timing {
    std::vector<double> times;
    double median = 0, min = 0, max = 0;
    std::vector<double> busy;  // per-thread seconds, mean over the timed runs
    double busy_mean = 0, busy_std = 0, busy_cv = 0, busy_max_over_mean = 0;
};

struct Snapshot {
    std::vector<double> xy;  // interleaved
    std::vector<uint8_t> state;
    size_t alive = 0, beached = 0, out = 0;

    template <class P> void take(const P& p) {
        const size_t n = p.size();
        xy.resize(2 * n);
        state.resize(n);
        alive = beached = out = 0;
        for (size_t i = 0; i < n; ++i) {
            xy[2 * i] = p.x(i);
            xy[2 * i + 1] = p.y(i);
            state[i] = p.state(i);
            (state[i] == ALIVE ? alive : state[i] == BEACHED ? beached : out)++;
        }
    }
};

struct Drift {
    bool valid = false;
    double max = 0, mean = 0;
    size_t state_mismatch = 0;
};

// Distance between two final positions: km on the sphere, domain units otherwise.
Drift compare(const std::vector<double>& a, const std::vector<double>& b, const std::vector<uint8_t>* sa,
              const std::vector<uint8_t>* sb, bool spherical) {
    if (a.size() != b.size()) throw std::runtime_error("drift check: particle counts differ");
    Drift d;
    d.valid = true;
    const size_t n = a.size() / 2;
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        double ex = a[2 * i] - b[2 * i], ey = a[2 * i + 1] - b[2 * i + 1];
        if (spherical) {
            ex *= std::cos(a[2 * i + 1] * kDegToRad) / kMetresToDeg / 1000.0;
            ey *= 1.0 / kMetresToDeg / 1000.0;
        }
        const double e = std::hypot(ex, ey);
        sum += e;
        d.max = std::max(d.max, e);
        if (sa && sb && (*sa)[i] != (*sb)[i]) ++d.state_mismatch;
    }
    d.mean = n ? sum / (double)n : 0;
    return d;
}

int configure_threads(const Cfg& c) {
#ifdef _OPENMP
    if (c.threads > 0) omp_set_num_threads(c.threads);
    omp_set_dynamic(0);
    const int chunk = c.chunk > 0 ? c.chunk : (c.schedule == "dynamic" ? 1024 : 0);
    omp_set_schedule(c.schedule == "static" ? omp_sched_static
                     : c.schedule == "dynamic" ? omp_sched_dynamic : omp_sched_guided, chunk);
    return omp_get_max_threads();
#else
    if (c.threads > 1)
        throw std::runtime_error("built without OpenMP: cannot run on " + std::to_string(c.threads) +
                                 " threads (see README, Build)");
    if (c.mode != "serial")
        std::cerr << "warning: built without OpenMP; the 'parallel' kernel runs on one thread\n";
    return 1;
#endif
}

int effective_chunk(const Cfg& c) { return c.chunk > 0 ? c.chunk : (c.schedule == "dynamic" ? 1024 : 0); }

template <class P, class Model>
Timing bench(const Cfg& c, const Model& m, long nsteps, bool parallel, int nthreads,
             const std::vector<double>& rx, const std::vector<double>& ry, Snapshot& last,
             TrajRecorder* traj) {
    Timing t;
    std::vector<double> busy_sum(nthreads, 0.0);
    P p;
    for (int r = 0; r < c.warmup + c.runs; ++r) {
        p.init(rx, ry);
        std::vector<double> busy((size_t)nthreads * kBusyStride, 0.0);
        const bool final_run = (r == c.warmup + c.runs - 1);
        const double secs = simulate(p, m, nsteps, parallel, busy, final_run ? traj : nullptr);
        if (r < c.warmup) continue;
        t.times.push_back(secs);
        for (int k = 0; k < nthreads; ++k) busy_sum[k] += busy[(size_t)k * kBusyStride];
    }
    last.take(p);

    std::vector<double> s = t.times;
    std::sort(s.begin(), s.end());
    t.min = s.front();
    t.max = s.back();
    t.median = (s.size() % 2) ? s[s.size() / 2] : 0.5 * (s[s.size() / 2 - 1] + s[s.size() / 2]);

    if (parallel) {
        double sum = 0, sq = 0, mx = 0;
        for (double b : busy_sum) {
            b /= c.runs;
            t.busy.push_back(b);
            sum += b;
            mx = std::max(mx, b);
        }
        t.busy_mean = sum / nthreads;
        for (double b : t.busy) sq += (b - t.busy_mean) * (b - t.busy_mean);
        t.busy_std = std::sqrt(sq / nthreads);
        t.busy_cv = t.busy_mean > 0 ? t.busy_std / t.busy_mean : 0;
        t.busy_max_over_mean = t.busy_mean > 0 ? mx / t.busy_mean : 0;
    }
    return t;
}

std::string join(const std::vector<double>& v) {
    std::ostringstream o;
    o.precision(9);
    for (size_t i = 0; i < v.size(); ++i) o << (i ? ";" : "") << v[i];
    return o.str();
}

void append_csv(const Cfg& c, const Meta& m, const char* mode, int threads, const Timing& t,
                const Snapshot& snap, const Drift& drift, double speedup, const Drift& truth) {
    if (c.csv.empty()) return;
    bool header = true;
    {
        std::ifstream in(c.csv);
        header = !in || in.peek() == std::ifstream::traits_type::eof();
    }
    std::ofstream f(c.csv, std::ios::app);
    if (!f) throw std::runtime_error("cannot write " + c.csv);
    if (header)
        f << "field,eps,nx,ny,nt,spherical,particles,steps,dt,release,mode,layout,schedule,chunk,threads,"
             "t_median,t_min,t_max,t_all,throughput,speedup,busy_mean,busy_std,busy_cv,"
             "busy_max_over_mean,busy_all,drift_max,drift_mean,state_mismatch,alive,beached,out,"
             "truth_err_mean,truth_err_max\n";
    const bool par = std::string(mode) == "parallel";
    auto opt = [](bool on, double v) {
        std::ostringstream o;
        o.precision(9);
        if (on) o << v;
        return o.str();
    };
    f.precision(9);
    f << m.field_name << ',' << m.eps << ',' << m.nx << ',' << m.ny << ',' << m.nt << ',' << m.spherical
      << ',' << c.particles << ',' << m.nsteps << ',' << m.dt << ',' << c.release << ',' << mode << ','
      << c.layout << ',' << (par ? c.schedule : "") << ',' << (par ? std::to_string(effective_chunk(c)) : "")
      << ',' << threads << ',' << t.median << ',' << t.min << ',' << t.max << ',' << join(t.times) << ','
      << (double)c.particles * (double)m.nsteps / t.median << ',' << opt(speedup > 0, speedup) << ','
      << opt(par, t.busy_mean) << ',' << opt(par, t.busy_std) << ',' << opt(par, t.busy_cv) << ','
      << opt(par, t.busy_max_over_mean) << ',' << (par ? join(t.busy) : "") << ','
      << opt(drift.valid, drift.max) << ',' << opt(drift.valid, drift.mean) << ','
      << (drift.valid ? std::to_string(drift.state_mismatch) : "") << ',' << snap.alive << ','
      << snap.beached << ',' << snap.out << ',' << opt(truth.valid, truth.mean) << ','
      << opt(truth.valid, truth.max) << '\n';
}

void write_outputs(const Cfg& c, const Meta& m, const Snapshot& snap, const std::vector<double>& rx,
                   const std::vector<double>& ry, const TrajRecorder& traj) {
    if (c.out.empty()) return;
    const size_t n = snap.state.size();
    write_npy(c.out + "_final.npy", snap.xy.data(), snap.xy.size() * 8, "<f8", {n, 2});
    write_npy(c.out + "_state.npy", snap.state.data(), n, "|u1", {n});
    std::vector<double> rel(2 * n);
    for (size_t i = 0; i < n; ++i) { rel[2 * i] = rx[i]; rel[2 * i + 1] = ry[i]; }
    write_npy(c.out + "_release.npy", rel.data(), rel.size() * 8, "<f8", {n, 2});
    if (traj.snapshots())
        write_npy(c.out + "_traj.npy", traj.xy.data(), traj.xy.size() * 4, "<f4",
                  {traj.snapshots(), traj.tracked, 2});
    if (m.grid)
        write_npy(c.out + "_land.npy", m.grid->land.data(), m.grid->land.size(), "|u1",
                  {(size_t)m.ny, (size_t)m.nx});

    std::ofstream j(c.out + "_meta.json");
    j.precision(12);
    j << "{\n  \"field\": \"" << m.field_name << "\",\n  \"eps\": " << m.eps << ",\n  \"spherical\": "
      << (m.spherical ? "true" : "false") << ",\n  \"nx\": " << m.nx << ",\n  \"ny\": " << m.ny
      << ",\n  \"nt\": " << m.nt << ",\n  \"extent\": [" << m.x0 << ", " << m.x1 << ", " << m.y0 << ", "
      << m.y1 << "],\n  \"dt\": " << m.dt << ",\n  \"steps\": " << m.nsteps << ",\n  \"particles\": " << n
      << ",\n  \"alive\": " << snap.alive << ",\n  \"beached\": " << snap.beached << ",\n  \"out\": "
      << snap.out << ",\n  \"traj_every\": " << traj.every << "\n}\n";
}

void print_timing(const char* label, const Cfg& c, const Meta& m, const Timing& t) {
    std::printf("%-9s median %.4f s  (min %.4f, max %.4f, %d runs)  %.3g particle-steps/s\n", label,
                t.median, t.min, t.max, c.runs, (double)c.particles * (double)m.nsteps / t.median);
}

template <class P, class Model> int execute(const Cfg& cfg, const Model& model, Meta meta) {
    Cfg c = cfg;
    const int nthreads = configure_threads(c);

    ReleaseBox box{meta.x0, meta.x1, meta.y0, meta.y1};
    if (!c.release_box.empty()) {
        if (std::sscanf(c.release_box.c_str(), "%lf,%lf,%lf,%lf", &box.x0, &box.x1, &box.y0, &box.y1) != 4)
            throw std::runtime_error("--release-box expects X0,X1,Y0,Y1");
    }
    std::vector<double> rx, ry;
    {
        double tau, dtau;
        const typename Model::Sampler s0 = model.sampler(0, tau, dtau);
        if (c.release_file.empty()) {
            seed_release(s0, box, c.particles, c.release == "lattice", c.seed, rx, ry);
        } else {
            const std::vector<double> pts = read_npy_f8(c.release_file);
            if (pts.empty() || pts.size() % 2) throw std::runtime_error("--release-file must hold (N,2) points");
            size_t bad = 0;
            for (size_t i = 0; i < pts.size(); i += 2) {
                double x = pts[i], y = pts[i + 1];
                if (s0.classify(x, y) != ALIVE) ++bad;  // also wraps x onto a periodic grid
                rx.push_back(x);
                ry.push_back(y);
            }
            if (bad)
                throw std::runtime_error(std::to_string(bad) + " of " + std::to_string(rx.size()) +
                                         " release points are on land or outside the domain");
            c.particles = rx.size();
            c.release = "file";
        }
    }

    std::printf("field     %s  grid %dx%d, %d slices%s\n", meta.field_name.c_str(), meta.nx, meta.ny,
                meta.nt, meta.grid ? "" : " (analytic)");
    std::printf("run       %zu particles, %ld steps of dt=%g, layout %s, release %s\n", c.particles,
                meta.nsteps, meta.dt, P::layout, c.release.c_str());

    TrajRecorder traj;
    if (!c.out.empty() && c.traj > 0)
        traj.setup(c.particles, c.traj, c.traj_every > 0 ? c.traj_every : std::max(1L, meta.nsteps / 200),
                   meta.nsteps);
    TrajRecorder* trajp = traj.every > 0 ? &traj : nullptr;

    Snapshot serial_snap, par_snap;
    Timing ts, tp;
    const bool do_serial = c.mode != "parallel", do_parallel = c.mode != "serial";

    if (do_serial) {
        ts = bench<P>(c, model, meta.nsteps, false, 1, rx, ry, serial_snap, do_parallel ? nullptr : trajp);
        print_timing("serial", c, meta, ts);
    }
    if (do_parallel) {
        tp = bench<P>(c, model, meta.nsteps, true, nthreads, rx, ry, par_snap, trajp);
        std::printf("threads   %d, schedule %s", nthreads, c.schedule.c_str());
        if (effective_chunk(c)) std::printf(",%d", effective_chunk(c));
        std::printf("\n");
        print_timing("parallel", c, meta, tp);
        std::printf("busy      mean %.4f s/thread, cv %.3f, max/mean %.3f\n", tp.busy_mean, tp.busy_cv,
                    tp.busy_max_over_mean);
    }
    const Snapshot& final_snap = do_parallel ? par_snap : serial_snap;

    // RQ4: drift of this run against the serial baseline (in-process or from --ref).
    Drift drift;
    const char* unit = meta.spherical ? "km" : "domain units";
    if (do_serial && do_parallel) {
        drift = compare(par_snap.xy, serial_snap.xy, &par_snap.state, &serial_snap.state, meta.spherical);
    } else if (!c.ref.empty()) {
        drift = compare(final_snap.xy, read_npy_f8(c.ref), nullptr, nullptr, meta.spherical);
    }
    double speedup = 0;
    if (do_serial && do_parallel) {
        speedup = ts.median / tp.median;
        std::printf("speedup   %.2fx on %d threads, efficiency %.1f%%\n", speedup, nthreads,
                    100.0 * speedup / nthreads);
    }
    if (drift.valid)
        std::printf("drift     vs serial: max %.3g, mean %.3g %s, %zu state mismatches\n", drift.max,
                    drift.mean, unit, drift.state_mismatch);

    // Double-gyre ground truth: closed-form field integrated with a 16x finer step.
    Drift truth;
    if (c.truth && c.field == "dg") {
        const int refine = 16;
        AnalyticModel ref{DoubleGyre{c.eps}, meta.dt / refine};
        ParticlesSoA p;
        p.init(rx, ry);
        std::vector<double> busy(kBusyStride, 0.0);
        simulate(p, ref, meta.nsteps * refine, false, busy);
        Snapshot ref_snap;
        ref_snap.take(p);
        truth = compare(final_snap.xy, ref_snap.xy, nullptr, nullptr, false);
        std::printf("truth     error vs analytic reference: mean %.3g, max %.3g domain units\n", truth.mean,
                    truth.max);
    }
    std::printf("fate      %zu afloat, %zu beached, %zu left the domain\n", final_snap.alive,
                final_snap.beached, final_snap.out);

    const Drift none;
    if (do_serial) append_csv(c, meta, "serial", 1, ts, serial_snap, none, 0, do_parallel ? none : truth);
    if (do_parallel) append_csv(c, meta, "parallel", nthreads, tp, par_snap, drift, speedup, truth);
    write_outputs(c, meta, final_snap, rx, ry, traj);
    return 0;
}

template <class Model> int dispatch(const Cfg& c, const Model& model, const Meta& meta) {
    return c.layout == "soa" ? execute<ParticlesSoA>(c, model, meta) : execute<ParticlesAoS>(c, model, meta);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Cfg c = parse(argc, argv);
        Meta meta;

        if (c.field == "dg") {
            if (c.dt <= 0) c.dt = 0.05;
            if (c.duration <= 0) c.duration = 20.0;
            meta.field_name = "double_gyre";
            meta.eps = c.eps;
            meta.x0 = 0; meta.x1 = 2; meta.y0 = 0; meta.y1 = 1;
            meta.dt = c.dt;
            meta.nsteps = std::max(1L, std::lround(c.duration / c.dt));
            if (c.analytic) return dispatch(c, AnalyticModel{DoubleGyre{c.eps}, c.dt}, meta);

            const GridField f = make_double_gyre_grid(c.eps, c.dg_nx, c.dg_ny, c.dg_slice_dt,
                                                      meta.nsteps * c.dt);
            meta.nx = f.nx; meta.ny = f.ny; meta.nt = f.nt; meta.grid = &f;
            return dispatch(c, GridModel{&f, c.dt}, meta);
        }

        const GridField f = load_netcdf_field(c.file);
        const double span = f.times.back();
        const double interval = f.times[1] - f.times[0];
        // Use a step that divides the slice spacing so no step straddles two brackets.
        if (c.dt <= 0) c.dt = 3600.0;
        c.dt = interval / std::ceil(interval / c.dt - 1e-9);
        double seconds = c.duration > 0 ? c.duration * 86400.0 : span;
        if (seconds > span) {
            std::cerr << "warning: --duration exceeds the " << span / 86400.0 << " days in the file; clamping\n";
            seconds = span;
        }
        size_t slash = c.file.find_last_of('/');
        meta.field_name = c.file.substr(slash == std::string::npos ? 0 : slash + 1);
        meta.spherical = f.spherical;
        meta.nx = f.nx; meta.ny = f.ny; meta.nt = f.nt; meta.grid = &f;
        meta.x0 = f.x0; meta.x1 = f.periodic_x ? f.x0 + f.nx * f.dx : f.xmax();
        meta.y0 = f.y0; meta.y1 = f.ymax();
        meta.dt = c.dt;
        meta.nsteps = std::max(1L, std::lround(seconds / c.dt));
        return dispatch(c, GridModel{&f, c.dt}, meta);
    } catch (const std::exception& e) {
        std::cerr << "mpd: " << e.what() << "\n";
        return 1;
    }
}
