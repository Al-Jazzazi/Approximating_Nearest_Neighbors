#include <iostream>
#include <vector>
#include "hnsw.h"

using namespace std;

int main() {
    // Load config
    Config* config = new Config();

    // Load nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** queries = new float*[config->num_queries];
    load_queries(config, nodes, queries);

    // Find and save actual nearest neighbors
    vector<vector<int>> actual_neighbors;
    knn_search(config, actual_neighbors, nodes, queries);
    save_ivecs(config->groundtruth_file, actual_neighbors);
}