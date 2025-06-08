# Graph-Based Nearest Neighbor Framework

This project provides a flexible framework for building, running, and benchmarking graph-based approximate nearest neighbor (ANN) search algorithms. It supports:

* Building and benchmarking **HNSW** graphs
* Pruning HNSW using **Grasp** (a cost-benefit optimization method)
* Building and benchmarking **Vamana** graphs
* Running and benchmarking general one-layer graph structures (e.g., **Efanna**, **NSG**)

The system is modular and configurable, allowing easy experimentation with various ANN strategies through centralized parameter settings.
---

## Project Structure

```
.
├── src/
│   ├── include/          # Header files, configs, and graph interfaces
│   ├── core/             # Graph algorithm implementations (HNSW, Grasp, Vamana, etc.)
│   ├── run/              # Entry points for running graph search
│   ├── benchmark/        # Parameterized benchmarking programs
│   └── tools/            # Utilities for dataset processing and analysis
├── build/                # Output binaries (auto-created)
├── runs/                 # (Placeholder) Can be used for run outputs
├── Makefile              # Build rules
└── README.md             # This file
```

---

## Build & Run

Each target is built individually and outputs a timestamped binary in `build/`. A symlink is also created for consistent access.

###  Example

```bash
make benchmark_slurm_graph
./build/benchmark_slurm_graph
```

This builds the executable from its dependencies and runs it through the symlink pointing to the most recent binary version.


## Executables & Targets

Below is a description of each target in the Makefile:

###  Run Graphs

These executables run a graph search instance using parameters from `config.h`.

| Target          | Description                                     |
| --------------- | ----------------------------------------------- |
| `run_hnsw`      | Runs HNSW or Grasp-based search                 |
| `run_vamana`    | Runs Vamana search                              |
| `run_graph`     | Runs the abstract graph (for Efanna, NSG, etc.) |
|  | Runs HNSW while logging every pop/push from datasets   |

### Benchmarks

These evaluate performance by varying a parameter (e.g. `k`) set in `config.h`. SLURM targets are meant to integrate with distributed runs.

| Target                   | Description                               |
| ------------------------ | ----------------------------------------- |
| `benchmark`              | General benchmark (HNSW/Grasp)            |
| `benchmark_grasp`        | Dedicated benchmark for Grasp             |
| `benchmark_slurm`        | SLURM-compatible benchmark for HNSW/Grasp |
| `benchmark_slurm_grasp`  | SLURM benchmark focused on Grasp          |
| `benchmark_slurm_vamana` | SLURM benchmark for Vamana                |
| `benchmark_slurm_graph`  | SLURM benchmark for abstract Graph        |
| `run_data_benchmark`     | Reruns every pop/push and measures time needed (need to run `run_data_type` first)     |

###  Dataset & Analysis Tools

| Target                 | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `generate_groundtruth` | Generates ground truth from a dataset               |
| `generate_training`    | Prepares training data (e.g., for pruning or optimization) |
| `dataset_metrics`      | Outputs basic statistics about the dataset                |
| `run_similarity`       | Compares the similiarty of nodes and queries set             |


Note: run_data_benchmark and run_data_benchmark are no longer used and were replaced by measuring real time accross an entire search when running any other benchmark. 
---

##  Build Details

Each target compiles a specific set of source files, outputs a timestamped `.out` binary in `build/`, and creates a symlink with a stable name:

Example from the Makefile:

```makefile
run_hnsw: src/run/run.cpp src/core/hnsw.cpp src/include/hnsw.h src/core/grasp.cpp ...
	$(CXX) $(CXXFLAGS) -o build/run_hnsw_$(EPOCH_TIME).out $^
	ln -sf run_hnsw_$(EPOCH_TIME).out build/run_hnsw
```

This ensures:

* You always access the latest binary via the symlink (`./build/run_hnsw`)
* Timestamped versions are preserved for reproducibility

---

## Configuration

All graph and benchmarking parameters are set in:

```
src/include/config.h
```

Change values in that file (e.g. `dataset`, num_return of edges, etc.) to experiment with different configurations. Then recompile and rerun the desired target.



##  TODO

* Save benchmark results in structured formats (e.g., CSV, SQLite, or other databases) instead of plain `.txt` files.

