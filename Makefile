CXX := g++
CXXFLAGS := -O2 -mavx

SRCS := $(wildcard HNSW/*.cpp GraSP/*.cpp)
OBJS := $(SRCS:.cpp=.o)
TARGETS := benchmark.out run_hnsw.out hnsw_save.out dataset_metrics.out dataset_comparison.out run_grasp.out

.PHONY: all clean

all: $(TARGETS)

benchmark.out: HNSW/benchmark.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

run_hnsw.out: HNSW/run_hnsw.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

hnsw_save.out: HNSW/hnsw_save.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

dataset_metrics.out: HNSW/dataset_metrics.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

dataset_comparison.out: HNSW/dataset_comparison.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

run_grasp.out: GraSP/run_grasp.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o $@ $^

clean:
	rm -f $(OBJS) $(TARGETS)