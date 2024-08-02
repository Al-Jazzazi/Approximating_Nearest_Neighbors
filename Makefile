CXX := g++
CXXFLAGS := -O2 -mavx -g

MAKE_DIRECTORIES := $(shell mkdir -p build runs)
EPOCH_TIME := $(shell date +%s)
SRCS := $(wildcard HNSW/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(SRCS))
COMMON_SRCS := config.h src/common.cpp src/common.h
TARGETS := run_hnsw run_vamana dataset_metrics generate_groundtruth benchmark benchmark_slurm
BUILD_PATH := build

.PHONY: all clean

all: $(TARGETS)

run_hnsw: src/run.cpp src/hnsw.cpp src/hnsw.h src/grasp.cpp src/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

run_vamana: src/vamana.cpp src/vamana.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

dataset_metrics: src/dataset_metrics.cpp src/hnsw.cpp src/hnsw.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

generate_groundtruth: src/generate_groundtruth.cpp src/hnsw.cpp src/hnsw.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

generate_training: src/generate_training.cpp src/hnsw.cpp src/hnsw.h src/grasp.cpp src/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

benchmark: src/benchmark.cpp src/hnsw.cpp src/hnsw.h src/grasp.cpp src/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@.out $^

benchmark_slurm: src/benchmark.cpp src/grasp.cpp src/grasp.h src/hnsw.cpp src/hnsw.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

clean:
	rm -f $(OBJS) $(TARGETS)