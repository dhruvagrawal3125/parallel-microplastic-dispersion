// Minimal reader for the NetCDF "classic" on-disk formats (CDF-1, CDF-2 64-bit
// offset, CDF-5 64-bit data). No external library: the header is parsed by hand
// and variables are read straight from the file as big-endian arrays.
//
// NetCDF-4/HDF5 files (what PO.DAAC and HYCOM serve) are not parsed here; they
// are subset and converted to classic format once by scripts/prep_currents.py.
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace mpd {

enum NcType : int {
    NC_BYTE = 1, NC_CHAR = 2, NC_SHORT = 3, NC_INT = 4, NC_FLOAT = 5, NC_DOUBLE = 6,
    NC_UBYTE = 7, NC_USHORT = 8, NC_UINT = 9, NC_INT64 = 10, NC_UINT64 = 11
};

struct NcAttr {
    std::string name;
    int type = 0;
    std::vector<double> num;  // numeric attributes
    std::string text;         // NC_CHAR attributes
};

struct NcDim {
    std::string name;
    uint64_t len = 0;  // for the record dimension this is numrecs
    bool unlimited = false;
};

struct NcVar {
    std::string name;
    std::vector<int> dimids;
    std::vector<uint64_t> shape;
    std::vector<NcAttr> atts;
    int type = 0;
    uint64_t vsize = 0;
    uint64_t begin = 0;
    bool record = false;

    const NcAttr* att(const std::string& n) const;
    uint64_t count() const;  // total number of elements
};

class NcFile {
public:
    explicit NcFile(const std::string& path);

    int version() const { return version_; }
    const std::vector<NcDim>& dims() const { return dims_; }
    const std::vector<NcVar>& vars() const { return vars_; }
    const std::vector<NcAttr>& global_atts() const { return gatts_; }

    const NcVar* var(const std::string& name) const;
    // First variable whose name matches one of `names`, or nullptr.
    const NcVar* find_var(const std::vector<std::string>& names) const;

    // Raw values converted to T (no scale_factor / add_offset applied).
    template <class T> void read(const NcVar& v, std::vector<T>& out) const;

private:
    std::string path_;
    mutable std::ifstream f_;
    int version_ = 0;
    uint64_t numrecs_ = 0;
    uint64_t recsize_ = 0;
    std::vector<NcDim> dims_;
    std::vector<NcVar> vars_;
    std::vector<NcAttr> gatts_;
};

}  // namespace mpd
