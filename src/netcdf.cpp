#include "mpd/netcdf.hpp"

#include <cstring>
#include <stdexcept>

namespace mpd {

namespace {

const uint32_t TAG_DIMENSION = 0x0A, TAG_VARIABLE = 0x0B, TAG_ATTRIBUTE = 0x0C;

size_t type_size(int t) {
    switch (t) {
        case NC_BYTE: case NC_CHAR: case NC_UBYTE: return 1;
        case NC_SHORT: case NC_USHORT: return 2;
        case NC_INT: case NC_FLOAT: case NC_UINT: return 4;
        case NC_DOUBLE: case NC_INT64: case NC_UINT64: return 8;
    }
    throw std::runtime_error("netcdf: unknown data type " + std::to_string(t));
}

// Decode one big-endian element of NetCDF type `t` at `p`.
template <class T> T decode(const unsigned char* p, int t) {
    switch (t) {
        case NC_BYTE: return (T)(int8_t)p[0];
        case NC_CHAR: case NC_UBYTE: return (T)p[0];
        case NC_SHORT: return (T)(int16_t)((p[0] << 8) | p[1]);
        case NC_USHORT: return (T)(uint16_t)((p[0] << 8) | p[1]);
        case NC_INT: case NC_UINT: case NC_FLOAT: {
            uint32_t b = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
            if (t == NC_INT) return (T)(int32_t)b;
            if (t == NC_UINT) return (T)b;
            float x;
            std::memcpy(&x, &b, 4);
            return (T)x;
        }
        default: {
            uint64_t b = 0;
            for (int i = 0; i < 8; ++i) b = (b << 8) | p[i];
            if (t == NC_INT64) return (T)(int64_t)b;
            if (t == NC_UINT64) return (T)b;
            double x;
            std::memcpy(&x, &b, 8);
            return (T)x;
        }
    }
}

struct Cursor {
    std::ifstream& f;
    int ver;

    void bytes(void* dst, size_t n) {
        f.read(static_cast<char*>(dst), (std::streamsize)n);
        if (!f) throw std::runtime_error("netcdf: unexpected end of file in header");
    }
    uint32_t u32() {
        unsigned char b[4];
        bytes(b, 4);
        return decode<uint32_t>(b, NC_UINT);
    }
    uint64_t u64() {
        unsigned char b[8];
        bytes(b, 8);
        return decode<uint64_t>(b, NC_UINT64);
    }
    // "NON_NEG" in the format spec: 32-bit, or 64-bit in CDF-5.
    uint64_t nn() { return ver == 5 ? u64() : u32(); }
    void pad(uint64_t n) {
        if (n % 4) f.seekg((std::streamoff)(4 - n % 4), std::ios::cur);
    }
    std::string name() {
        uint64_t n = nn();
        std::string s(n, '\0');
        if (n) bytes(&s[0], n);
        pad(n);
        return s;
    }
    // Reads a list header and returns its element count; `tag` must match unless absent.
    uint64_t list(uint32_t tag) {
        uint32_t t = u32();
        uint64_t n = nn();
        if (t == 0 && n == 0) return 0;
        if (t != tag) throw std::runtime_error("netcdf: malformed header (unexpected list tag)");
        return n;
    }
    std::vector<NcAttr> attrs() {
        std::vector<NcAttr> out(list(TAG_ATTRIBUTE));
        for (auto& a : out) {
            a.name = name();
            a.type = (int)u32();
            uint64_t n = nn();
            size_t sz = type_size(a.type);
            std::vector<unsigned char> buf(n * sz);
            if (!buf.empty()) bytes(buf.data(), buf.size());
            pad(n * sz);
            if (a.type == NC_CHAR) {
                a.text.assign(buf.begin(), buf.end());
                while (!a.text.empty() && a.text.back() == '\0') a.text.pop_back();
            } else {
                a.num.resize(n);
                for (uint64_t i = 0; i < n; ++i) a.num[i] = decode<double>(&buf[i * sz], a.type);
            }
        }
        return out;
    }
};

}  // namespace

const NcAttr* NcVar::att(const std::string& n) const {
    for (const auto& a : atts)
        if (a.name == n) return &a;
    return nullptr;
}

uint64_t NcVar::count() const {
    uint64_t n = 1;
    for (uint64_t s : shape) n *= s;
    return n;
}

NcFile::NcFile(const std::string& path) : path_(path), f_(path, std::ios::binary) {
    if (!f_) throw std::runtime_error("netcdf: cannot open " + path);
    unsigned char magic[4] = {0, 0, 0, 0};
    f_.read((char*)magic, 4);
    if (!f_) throw std::runtime_error("netcdf: file too short: " + path);
    if (magic[0] == 0x89 && magic[1] == 'H' && magic[2] == 'D' && magic[3] == 'F')
        throw std::runtime_error("netcdf: " + path +
                                 " is NetCDF-4/HDF5; convert it with scripts/prep_currents.py first");
    if (magic[0] != 'C' || magic[1] != 'D' || magic[2] != 'F' ||
        (magic[3] != 1 && magic[3] != 2 && magic[3] != 5))
        throw std::runtime_error("netcdf: " + path + " is not a classic NetCDF file");
    version_ = magic[3];

    Cursor c{f_, version_};
    numrecs_ = c.nn();

    dims_.resize(c.list(TAG_DIMENSION));
    for (auto& d : dims_) {
        d.name = c.name();
        d.len = c.nn();
        d.unlimited = (d.len == 0);
    }
    gatts_ = c.attrs();

    vars_.resize(c.list(TAG_VARIABLE));
    for (auto& v : vars_) {
        v.name = c.name();
        uint64_t nd = c.nn();
        v.dimids.resize(nd);
        for (auto& id : v.dimids) {
            id = (int)c.nn();
            if (id < 0 || (size_t)id >= dims_.size())
                throw std::runtime_error("netcdf: bad dimension id in variable " + v.name);
        }
        v.atts = c.attrs();
        v.type = (int)c.u32();
        v.vsize = c.nn();
        v.begin = (version_ == 1) ? c.u32() : c.u64();
        v.record = !v.dimids.empty() && dims_[v.dimids[0]].unlimited;
    }

    // Streaming writers leave numrecs = 0xFFFFFFFF; derive it from the file size.
    int nrecvars = 0;
    for (const auto& v : vars_)
        if (v.record) { ++nrecvars; recsize_ += v.vsize; }
    if (nrecvars == 1) {
        // A lone record variable is stored without per-record padding.
        for (const auto& v : vars_)
            if (v.record) {
                uint64_t n = type_size(v.type);
                for (size_t k = 1; k < v.dimids.size(); ++k) n *= dims_[v.dimids[k]].len;
                recsize_ = n;
            }
    }
    if (version_ != 5 && numrecs_ == 0xFFFFFFFFull && recsize_ > 0) {
        f_.seekg(0, std::ios::end);
        uint64_t size = (uint64_t)f_.tellg(), first = size;
        for (const auto& v : vars_)
            if (v.record && v.begin < first) first = v.begin;
        numrecs_ = (size - first) / recsize_;
    }
    for (auto& d : dims_)
        if (d.unlimited) d.len = numrecs_;
    for (auto& v : vars_)
        for (int id : v.dimids) v.shape.push_back(dims_[id].len);
}

const NcVar* NcFile::var(const std::string& name) const {
    for (const auto& v : vars_)
        if (v.name == name) return &v;
    return nullptr;
}

const NcVar* NcFile::find_var(const std::vector<std::string>& names) const {
    for (const auto& n : names)
        if (const NcVar* v = var(n)) return v;
    return nullptr;
}

template <class T> void NcFile::read(const NcVar& v, std::vector<T>& out) const {
    const size_t sz = type_size(v.type);
    const uint64_t total = v.count();
    out.resize(total);
    if (total == 0) return;

    // Record variables are interleaved record by record; fixed ones are one block.
    const uint64_t nblocks = v.record ? v.shape[0] : 1;
    const uint64_t per_block = total / nblocks;
    std::vector<unsigned char> buf(per_block * sz);
    for (uint64_t b = 0; b < nblocks; ++b) {
        f_.clear();
        f_.seekg((std::streamoff)(v.begin + (v.record ? b * recsize_ : 0)));
        f_.read((char*)buf.data(), (std::streamsize)buf.size());
        if (!f_) throw std::runtime_error("netcdf: short read of variable " + v.name + " in " + path_);
        T* dst = out.data() + b * per_block;
        for (uint64_t i = 0; i < per_block; ++i) dst[i] = decode<T>(&buf[i * sz], v.type);
    }
}

template void NcFile::read<float>(const NcVar&, std::vector<float>&) const;
template void NcFile::read<double>(const NcVar&, std::vector<double>&) const;

}  // namespace mpd
