CXX := g++
CXXFLAGS := -O2 -mavx

SRCS := $(wildcard HNSW/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(SRCS))
TARGETS := benchmark run save dataset_metrics benchmark_slurm
BUILD_PATH := build
MAKE_DIRECTORIES := $(shell mkdir -p build runs)
EPOCH_TIME := $(shell date +%s)

.PHONY: all clean

all: $(TARGETS)

run: HNSW/run.cpp HNSW/hnsw.cpp HNSW/hnsw.h HNSW/grasp.cpp HNSW/grasp.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

save: HNSW/save.cpp HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

dataset_metrics: HNSW/dataset_metrics.cpp HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

benchmark: HNSW/benchmark.cpp HNSW/hnsw.cpp HNSW/hnsw.h HNSW/grasp.cpp HNSW/grasp.h config.h
	$(CXX) $(CXXFLAGS) -g -pg -o ${BUILD_PATH}/$@.out $^

benchmark_slurm: HNSW/benchmark.cpp HNSW/grasp.cpp HNSW/grasp.h HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

clean:
	rm -f $(OBJS) $(TARGETS)