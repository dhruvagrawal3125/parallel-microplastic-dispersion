// roofline: measures the two platform numbers the RQ2 roofline needs --
// sustained memory bandwidth (STREAM triad) and peak in-cache floating-point
// rate -- per thread count, and places the advection kernel on that roofline
// using a hand count of its arithmetic intensity.
//
//   roofline [--threads 1,2,4,8] [--mb 512] [--csv results/roofline.csv]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// STREAM triad a = b + s*c over arrays far larger than cache. Returns GB/s,
// counting 24 bytes per element (two reads, one write; no write-allocate term).
double triad_gbs(size_t n, int threads) {
    (void)threads;
    std::vector<double> a(n), b(n), c(n);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)n; ++i) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }
    double best = 1e30;
    for (int rep = 0; rep < 6; ++rep) {
        const double t0 = now();
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)n; ++i) a[i] = b[i] + 3.0 * c[i];
        best = std::min(best, now() - t0);
    }
    volatile double sink = a[n / 2];
    (void)sink;
    return 24.0 * (double)n / best / 1e9;
}

// Peak rate: a + s*b on arrays that fit in L1, two flops per element. Returns GFLOP/s.
double peak_gflops(int threads) {
    const size_t n = 1024;
    const long reps = 400000;
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now();
        #pragma omp parallel num_threads(threads)
        {
            std::vector<double> a(n, 1.0), b(n, 1e-9);
            for (long r = 0; r < reps; ++r) {
                for (size_t i = 0; i < n; ++i) a[i] = a[i] + 1.0000001 * b[i];
                if (a[r % n] < 0) std::abort();  // keeps the loop from being optimised away
            }
        }
        best = std::min(best, now() - t0);
    }
    return 2.0 * (double)n * (double)reps * threads / best / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    std::string list, csv;
    size_t mb = 512;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--threads") list = argv[i + 1];
        else if (k == "--mb") mb = (size_t)std::atol(argv[i + 1]);
        else if (k == "--csv") csv = argv[i + 1];
    }
    int maxt = 1;
#ifdef _OPENMP
    maxt = omp_get_max_threads();
#endif
    std::vector<int> threads;
    if (list.empty()) {
        for (int t = 1; t < maxt; t *= 2) threads.push_back(t);
        threads.push_back(maxt);
    } else {
        std::stringstream ss(list);
        for (std::string tok; std::getline(ss, tok, ',');) threads.push_back(std::atoi(tok.c_str()));
    }
#ifndef _OPENMP
    std::printf("built without OpenMP: single-thread numbers only\n");
    threads.assign(1, 1);
#endif

    // Kernel arithmetic intensity, counted from rk4_step + GridSampler::vel:
    // one velocity lookup is ~50 flops (index maths, 4 weights, 4 bilinear sums,
    // 2 time lerps) plus a cosine (~20) on the sphere; RK4 adds ~25 flops around
    // four lookups. Bytes: 16 B of position read + 16 B written + 1 B state per
    // particle-step always stream through memory; the 256 B of field reads
    // (4 corners x 2 components x 2 slices x 4 stages x 4 B) only cost DRAM
    // traffic when the bracketing slices do not fit in cache.
    const double flops = 4 * 70.0 + 25.0;
    const double bytes_cached = 33.0, bytes_uncached = 33.0 + 256.0;
    const double ai_hi = flops / bytes_cached, ai_lo = flops / bytes_uncached;
    std::printf("kernel arithmetic intensity (estimate): %.2f flop/byte with the field in cache, "
                "%.2f if every field read misses\n\n", ai_hi, ai_lo);

    const size_t n = mb * 1024 * 1024 / 24;
    std::printf("%8s %12s %14s %22s %22s\n", "threads", "triad GB/s", "peak GFLOP/s",
                "bound, field cached", "bound, field uncached");
    std::ofstream out;
    if (!csv.empty()) {
        out.open(csv);
        out << "threads,bandwidth_gbs,peak_gflops,ai_cached,ai_uncached,bound_cached_gflops,"
               "bound_uncached_gflops,flops_per_particle_step\n";
    }
    for (int t : threads) {
        const double bw = triad_gbs(n, t), peak = peak_gflops(t);
        const double b_hi = std::min(peak, ai_hi * bw), b_lo = std::min(peak, ai_lo * bw);
        std::printf("%8d %12.1f %14.1f %15.1f GFLOP/s %15.1f GFLOP/s\n", t, bw, peak, b_hi, b_lo);
        if (out) out << t << ',' << bw << ',' << peak << ',' << ai_hi << ',' << ai_lo << ',' << b_hi << ','
                     << b_lo << ',' << flops << '\n';
    }
    std::printf("\nDivide a bound by %.0f flops to get the particle-steps/s ceiling to compare with mpd's "
                "throughput.\n", flops);
    return 0;
}
