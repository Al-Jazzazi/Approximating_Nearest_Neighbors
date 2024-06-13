CXX := g++
CXXFLAGS := -O2 -mavx

SRCS := $(wildcard HNSW/*.cpp Grasp/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(SRCS))
TARGETS := benchmark run_hnsw hnsw_save dataset_metrics run_grasp
BUILD_PATH := build
MAKE_DIRECTORIES := $(shell mkdir -p build)

.PHONY: all clean

all: $(TARGETS)

benchmark: HNSW/benchmark.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

run_hnsw: HNSW/run_hnsw.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

hnsw_save: HNSW/hnsw_save.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

dataset_metrics: HNSW/dataset_metrics.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

run_grasp: Grasp/run_grasp.cpp HNSW/hnsw.cpp HNSW/hnsw.h
	$(CXX) $(CXXFLAGS) -g -o ${BUILD_PATH}/$@.out $^

clean:
	rm -f $(OBJS) $(TARGETS)