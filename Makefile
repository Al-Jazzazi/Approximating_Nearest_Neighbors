CXX := g++
CXXFLAGS := -mavx -g  

MAKE_DIRECTORIES := $(shell mkdir -p build runs)
EPOCH_TIME := $(shell date +%s)
SRCS := $(wildcard src/*.cpp src/core/*.cpp src/benchmark/*.cpp src/run/*cpp src/tools/*cpp)
OBJS := $(SRCS:.cpp=.o)
COMMON_SRCS := src/include/config.h src/core/utils.cpp src/include/utils.h
TARGETS := run_hnsw run_vamana run_graph dataset_metrics generate_groundtruth generate_training run_similarity benchmark benchmark_slurm benchmark_grasp benchmark_slurm_grasp benchmark_slurm_vamana benchmark_slurm_graph run_data_type run_data_benchmark
BUILD_PATH := build

.PHONY: all clean

all: $(TARGETS)

run_hnsw: src/run/run.cpp src/core/hnsw.cpp src/include/hnsw.h src/core/grasp.cpp src/include/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@


run_vamana: src/run/run_vamana.cpp src/core/vamana.cpp src/include/vamana.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

run_graph: src/run/run_graph.cpp src/core/graph.cpp src/include/graph.h $(COMMON_SRCS) 
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

dataset_metrics: src/tools/dataset_metrics.cpp src/core/hnsw.cpp src/include/hnsw.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

generate_groundtruth: src/tools/generate_groundtruth.cpp src/core/hnsw.cpp src/include/hnsw.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

generate_training: src/tools/generate_training.cpp src/core/hnsw.cpp src/include/hnsw.h src/core/grasp.cpp src/include/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

run_similarity: src/tools/compare_similarity.cpp  $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@
benchmark: src/benchmark/benchmark.cpp src/core/hnsw.cpp src/include/hnsw.h src/core/grasp.cpp src/include/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@.out $^

benchmark_slurm: src/benchmark/benchmark.cpp src/core/hnsw.cpp src/include/hnsw.h src/core/grasp.cpp src/include/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

benchmark_grasp: src/benchmark/benchmark.cpp src/core/hnsw.cpp src/include/hnsw.h src/core/grasp.cpp src/include/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@.out $^

benchmark_slurm_grasp: src/benchmark/benchmark.cpp src/core/hnsw.cpp src/include/hnsw.h src/core/grasp.cpp src/include/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

benchmark_slurm_vamana: src/benchmark/benchmark_vamana.cpp src/core/vamana.cpp src/include/vamana.h  $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@
benchmark_slurm_graph: src/benchmark/graph_benchmark.cpp  src/core/graph.cpp src/include/graph.h  $(COMMON_SRCS) 
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@
	
run_data_type: src/tools/data_type_metrics.cpp src/core/hnsw.cpp src/include/hnsw.h src/core/grasp.cpp src/include/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@
run_data_benchmark: src/benchmark/benchmark_data_types.cpp src/core/hnsw.cpp src/include/hnsw.h src/core/grasp.cpp src/include/grasp.h $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o ${BUILD_PATH}/$@_$(EPOCH_TIME).out $^
	ln -sf $@_$(EPOCH_TIME).out  ${BUILD_PATH}/$@

clean:
	rm -f $(OBJS) $(TARGETS)
