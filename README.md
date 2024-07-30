# Summer2024-Research

This repository builds off the HNSW/Vamana research project conducted by Kenny Heung and
Silvia Wei in summer 2023: https://github.com/t-h24/Summer2023-Research

## Setup
Run the following to download the SIFT1M, Deep1M, GIST1M, and GloVe datasets: `source download_datasets.bash`

If you want to use the Deep or GloVe datasets, you should also run `make generate_groundtruth && ./build/generate_groundtruth.out` with the following config.h arguments:
- dataset = "glove" (or "deep")
- num_return = 100
- num_nodes = 1000000

This will generate groundtruth files, which future runs can compare against to evaluate recall. If you want to use num_return values greater than 100 in future runs, set the above num_return accordingly.

## Usage
Run: `make <target> && ./build/<target>`

For example, `make run && ./build/run`