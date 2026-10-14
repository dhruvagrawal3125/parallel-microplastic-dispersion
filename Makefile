# Parallel microplastic dispersion: build, test, benchmark.
#
#   make            build build/mpd and build/roofline
#   make test       build and run the test suite
#   make OPENMP=0   force a serial-only build
#
# Linux (g++/clang++) picks up OpenMP with -fopenmp. Apple clang ships without
# the OpenMP runtime, so on macOS the build looks for libomp under
# LIBOMP_PREFIX (default: third_party/libomp, then Homebrew).

CXX      ?= c++
OPT      ?= -O3 -march=native
CXXFLAGS += -std=c++17 -Wall -Wextra -Iinclude $(OPT)
OPENMP   ?= 1
BUILD    := build

ifeq ($(OPENMP),1)
  ifeq ($(shell uname -s),Darwin)
    LIBOMP_PREFIX ?= $(firstword $(wildcard third_party/libomp /opt/homebrew/opt/libomp /usr/local/opt/libomp))
    ifneq ($(LIBOMP_PREFIX),)
      OMPFLAGS ?= -Xclang -fopenmp -I$(LIBOMP_PREFIX)/include
      OMPLIBS  ?= -L$(LIBOMP_PREFIX)/lib -lomp -Wl,-rpath,$(abspath $(LIBOMP_PREFIX))/lib
    else
      $(warning libomp not found: building WITHOUT OpenMP (serial only). Set LIBOMP_PREFIX=/path/to/libomp)
    endif
  else
    OMPFLAGS ?= -fopenmp
    OMPLIBS  ?= -fopenmp
  endif
endif

HEADERS := $(wildcard include/mpd/*.hpp)
LIBOBJS := $(BUILD)/netcdf.o $(BUILD)/field.o

all: $(BUILD)/mpd $(BUILD)/roofline

$(BUILD)/%.o: src/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

$(BUILD)/mpd: $(BUILD)/main.o $(LIBOBJS)
	$(CXX) $^ $(OMPLIBS) -o $@

$(BUILD)/roofline: tools/roofline.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< $(OMPLIBS) -o $@

$(BUILD)/mpd_tests: tests/test_main.cpp $(LIBOBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) tests/test_main.cpp $(LIBOBJS) $(OMPLIBS) -o $@

$(BUILD):
	mkdir -p $(BUILD)

test: $(BUILD)/mpd_tests
	python3 scripts/make_test_nc.py $(BUILD)/testdata
	$(BUILD)/mpd_tests $(BUILD)/testdata

clean:
	rm -rf $(BUILD)

data: data/demo_eddies.nc

data/demo_eddies.nc:
	mkdir -p data
	python3 scripts/make_test_nc.py data

.PHONY: all test clean data
