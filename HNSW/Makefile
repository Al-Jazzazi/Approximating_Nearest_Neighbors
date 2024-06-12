CXX := g++
CXXFLAGS := -O2 -mavx

SRCS := hnsw.cpp run_hnsw.cpp beam_benchmark.cpp benchmark.cpp hnsw_save.cpp dataset_metrics.cpp dataset_comparison.cpp
OBJS := $(SRCS:.cpp=.o)
TARGETS := beam_benchmark.out benchmark.out run_hnsw.out hnsw_save.out dataset_metrics.out dataset_comparison.out

.PHONY: all clean

all: $(TARGETS)

beam_benchmark.out: beam_benchmark.cpp hnsw.cpp hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

benchmark.out: benchmark.cpp hnsw.cpp hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

run_hnsw.out: run_hnsw.cpp hnsw.cpp hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

hnsw_save.out: hnsw_save.cpp hnsw.cpp hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

dataset_metrics.out: dataset_metrics.cpp hnsw.cpp hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

dataset_comparison.out: dataset_comparison.cpp hnsw.cpp hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

clean:
	rm -f $(OBJS) $(TARGETS)