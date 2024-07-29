CXX := g++
CXXFLAGS := -O2 -mavx -g

MAKE_DIRECTORIES := $(shell mkdir -p build runs)
EPOCH_TIME := $(shell date +%s)
SRCS := $(wildcard HNSW/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(SRCS))
TARGETS := run dataset_metrics generate_groundtruth benchmark benchmark_slurm
BUILD_PATH := build

.PHONY: all clean

all: $(TARGETS)

run: HNSW/run.cpp HNSW/hnsw.cpp HNSW/hnsw.h HNSW/grasp.cpp HNSW/grasp.h config.h
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@.out $^

dataset_metrics: HNSW/dataset_metrics.cpp HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@.out $^

generate_groundtruth: HNSW/generate_groundtruth.cpp HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@.out $^

generate_training: HNSW/generate_training.cpp HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@.out $^

benchmark: HNSW/benchmark.cpp HNSW/hnsw.cpp HNSW/hnsw.h HNSW/grasp.cpp HNSW/grasp.h config.h
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@.out $^

benchmark_slurm: HNSW/benchmark.cpp HNSW/grasp.cpp HNSW/grasp.h HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

clean:
	rm -f $(OBJS) $(TARGETS)