# Summer2024-Research

This repository builds off the HNSW/Vamana research project conducted by Kenny Heung and
Silvia Wei in summer 2023: https://github.com/t-h24/Summer2023-Research

## Setup
1. Download the SIFT1M, Deep1M, GIST1M, and GloVe datasets with `source download_datasets.bash`
2. Run `make generate_groundtruth && ./build/generate_groundtruth.out` with the following config.h variables
   - dataset = "sift"
   - num_return = 100
   - num_nodes = 1000000
   
   This will generate the groundtruth file for SIFT1M (which is slightly different from the downloaded groundtruth). Repeat this for the other datasets.
3. Now, you can run the other C++ files with `make <target> && ./build/<target>`

   - Example: `make run && ./build/run`