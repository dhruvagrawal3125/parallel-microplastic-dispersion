// Particle storage in the two layouts compared under RQ3, plus release-point seeding.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "mpd/field.hpp"

namespace mpd {

// Structure-of-arrays: one contiguous array per attribute.
struct ParticlesSoA {
    static constexpr const char* layout = "soa";
    std::vector<double> xs, ys, x0s, y0s;
    std::vector<uint32_t> ages, ids;
    std::vector<uint8_t> states;

    void init(const std::vector<double>& rx, const std::vector<double>& ry) {
        const size_t n = rx.size();
        xs = rx; ys = ry; x0s = rx; y0s = ry;
        ages.assign(n, 0);
        states.assign(n, ALIVE);
        ids.resize(n);
        for (size_t i = 0; i < n; ++i) ids[i] = (uint32_t)i;
    }
    size_t size() const { return xs.size(); }
    double& x(size_t i) { return xs[i]; }
    double& y(size_t i) { return ys[i]; }
    uint8_t& state(size_t i) { return states[i]; }
    uint32_t& age(size_t i) { return ages[i]; }
    double x(size_t i) const { return xs[i]; }
    double y(size_t i) const { return ys[i]; }
    uint8_t state(size_t i) const { return states[i]; }
};

// Array-of-structs: the same attributes interleaved per particle.
struct ParticlesAoS {
    static constexpr const char* layout = "aos";
    struct Rec {
        double x, y, x0, y0;
        uint32_t age, id;
        uint8_t state;
    };
    std::vector<Rec> a;

    void init(const std::vector<double>& rx, const std::vector<double>& ry) {
        a.resize(rx.size());
        for (size_t i = 0; i < a.size(); ++i) a[i] = Rec{rx[i], ry[i], rx[i], ry[i], 0, (uint32_t)i, ALIVE};
    }
    size_t size() const { return a.size(); }
    double& x(size_t i) { return a[i].x; }
    double& y(size_t i) { return a[i].y; }
    uint8_t& state(size_t i) { return a[i].state; }
    uint32_t& age(size_t i) { return a[i].age; }
    double x(size_t i) const { return a[i].x; }
    double y(size_t i) const { return a[i].y; }
    uint8_t state(size_t i) const { return a[i].state; }
};

struct ReleaseBox {
    double x0, x1, y0, y1;
};

// Deterministic release points: the same (box, count, mode, seed) always gives
// the same particles in the same order. Points on land or outside the domain
// are skipped. `lattice` orders particles row by row, so neighbours in memory
// are neighbours in space; `random` scatters them.
template <class Sampler>
void seed_release(const Sampler& s, const ReleaseBox& box, size_t count, bool lattice, uint64_t seed,
                  std::vector<double>& rx, std::vector<double>& ry) {
    rx.clear();
    ry.clear();
    if (count == 0) return;
    const double w = box.x1 - box.x0, h = box.y1 - box.y0;
    if (!(w > 0 && h > 0)) throw std::runtime_error("release box is empty");

    if (!lattice) {
        uint64_t state = seed;
        auto next = [&state]() {  // splitmix64
            uint64_t z = (state += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return (double)((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);
        };
        size_t tries = 0;
        while (rx.size() < count) {
            double x = box.x0 + w * next(), y = box.y0 + h * next();
            if (++tries > 1000 * count + 100000) throw std::runtime_error("release box is almost all land");
            if (s.classify(x, y) == ALIVE) { rx.push_back(x); ry.push_back(y); }
        }
        return;
    }

    // Lattice: grow the point count until enough land-free points exist, then
    // thin evenly down to exactly `count`.
    double target = (double)count;
    for (int attempt = 0; attempt < 40; ++attempt) {
        const size_t ncol = (size_t)std::fmax(1.0, std::ceil(std::sqrt(target * w / h)));
        const size_t nrow = (size_t)std::fmax(1.0, std::ceil(target / (double)ncol));
        std::vector<double> cx, cy;
        cx.reserve(ncol * nrow);
        cy.reserve(ncol * nrow);
        for (size_t r = 0; r < nrow; ++r)
            for (size_t c = 0; c < ncol; ++c) {
                double x = box.x0 + w * ((double)c + 0.5) / (double)ncol;
                double y = box.y0 + h * ((double)r + 0.5) / (double)nrow;
                if (s.classify(x, y) == ALIVE) { cx.push_back(x); cy.push_back(y); }
            }
        if (cx.size() >= count) {
            rx.resize(count);
            ry.resize(count);
            for (size_t k = 0; k < count; ++k) {
                const size_t src = (size_t)((double)k * (double)cx.size() / (double)count);
                rx[k] = cx[src];
                ry[k] = cy[src];
            }
            return;
        }
        const double frac = cx.empty() ? 0.05 : (double)cx.size() / (double)(ncol * nrow);
        target = std::fmax(target * 1.1, (double)count / frac * 1.02);
    }
    throw std::runtime_error("release box is almost all land");
}

}  // namespace mpd
