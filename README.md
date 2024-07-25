# Summer2024-Research

This repository builds off the HNSW/Vamana research project conducted by Kenny Heung and
Silvia Wei in summer 2023: https://github.com/t-h24/Summer2023-Research

## Setup
Run the following to download the SIFT1M, Deep1M, GIST1M, and GloVe datasets: `source download_datasets.bash`

If you want to use the GloVe dataset, you should also run `make run && ./build/run.out` with the following config.h arguments:
- dataset = "glove"
- num_return = 100
- num_nodes = 1000000
- export_groundtruth = true 

This will generate a groundtruth file for GloVe, which will make future runs faster.

## Usage
Run: `make <target> && ./build/<target>.out`

For example, `make && ./build/run_hnsw.out`