CXX := g++
CXXFLAGS := -O2 -mavx

SRCS := $(wildcard HNSW/*.cpp Grasp/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(SRCS))
TARGETS := benchmark_hnsw run_hnsw save_hnsw dataset_metrics benchmark_grasp run_grasp
BUILD_PATH := build
MAKE_DIRECTORIES := $(shell mkdir -p build runs)

.PHONY: all clean

all: $(TARGETS)

benchmark_hnsw: HNSW/benchmark.cpp HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

run_hnsw: HNSW/run.cpp HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

save_hnsw: HNSW/save_hnsw.cpp HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

dataset_metrics: HNSW/dataset_metrics.cpp HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

benchmark_grasp: Grasp/benchmark.cpp Grasp/grasp.cpp Grasp/grasp.h HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^
	
run_grasp: Grasp/run.cpp Grasp/grasp.cpp Grasp/grasp.h HNSW/hnsw.cpp HNSW/hnsw.h config.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^


clean:
	rm -f $(OBJS) $(TARGETS)