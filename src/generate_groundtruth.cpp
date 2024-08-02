#include <iostream>
#include <vector>
#include "hnsw.h"

int main() {
    // Load config
    Config* config = new Config();

    // Load nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** queries = new float*[config->num_queries];
    load_queries(config, nodes, queries);

    vector<vector<int>> actual_neighbors;
    knn_search(config, actual_neighbors, nodes, queries);
    save_ivecs(config->groundtruth_file, actual_neighbors);
}