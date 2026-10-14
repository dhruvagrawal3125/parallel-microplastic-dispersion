// Tiny NumPy .npy writer/reader so results load directly with numpy.load.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mpd {

// descr is the NumPy dtype string, e.g. "<f8", "<f4", "|u1".
inline void write_npy(const std::string& path, const void* data, size_t bytes, const char* descr,
                      const std::vector<size_t>& shape) {
    std::string dims;
    for (size_t s : shape) dims += std::to_string(s) + ",";
    std::string hdr = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" + dims + "), }";
    while ((10 + hdr.size() + 1) % 64) hdr += ' ';
    hdr += '\n';
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    const uint16_t len = (uint16_t)hdr.size();
    const unsigned char lenb[2] = {(unsigned char)(len & 0xFF), (unsigned char)(len >> 8)};
    f.write("\x93NUMPY\x01\x00", 8);
    f.write((const char*)lenb, 2);
    f.write(hdr.data(), (std::streamsize)hdr.size());
    f.write((const char*)data, (std::streamsize)bytes);
    if (!f) throw std::runtime_error("write failed: " + path);
}

// Reads the payload of a "<f8" .npy written by write_npy.
inline std::vector<double> read_npy_f8(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    unsigned char pre[10];
    f.read((char*)pre, 10);
    if (!f || pre[0] != 0x93 || pre[1] != 'N') throw std::runtime_error(path + " is not a .npy file");
    const size_t len = pre[8] | ((size_t)pre[9] << 8);
    std::string hdr(len, '\0');
    f.read(&hdr[0], (std::streamsize)len);
    if (hdr.find("'<f8'") == std::string::npos) throw std::runtime_error(path + " is not float64");
    const std::streamoff start = f.tellg();
    f.seekg(0, std::ios::end);
    const size_t n = (size_t)(f.tellg() - start) / 8;
    f.seekg(start);
    std::vector<double> out(n);
    f.read((char*)out.data(), (std::streamsize)(n * 8));
    return out;
}

}  // namespace mpd
