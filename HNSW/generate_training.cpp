#include <iostream>
#include <vector>
#include "hnsw.h"

int main() {
    // Load config
    Config* config = new Config();
    HNSW* hnsw = NULL;
    config->num_return = 100;

    // Load nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** training = nullptr; 
    training = new float*[config->num_training];

    // Create HNSW graph 
    if (config->load_graph_file) {
        hnsw = init_hnsw(config, nodes);
        load_hnsw_files(config, hnsw, nodes, true);
    } else {
        hnsw = init_hnsw(config, nodes);
        for (int i = 1; i < config->num_nodes; ++i) {
            hnsw->insert(config, i);
        }
    }

    mt19937 gen(config->graph_seed);
    uniform_real_distribution<float> dis(0, 0.9999999);
    for (int i = 0; i< config->num_training; i++){
        // Choose a random node out of the source dataset
        training[i] = new float[config->dimensions];
        int index_first = dis(gen) * config->num_nodes;
        pair<int, float*> query = make_pair(index_first, nodes[index_first]);

        // Choose 2 random nearest neighbors out of top 100
        vector<Edge*> path;
        vector<pair<float, int>> nearest_neighbors = hnsw->nn_search(config, path, query, config->num_return, true);
        int index_second =  nearest_neighbors[static_cast<int>(dis(gen) * 100)].second;
        int index_third = nearest_neighbors[static_cast<int>(dis(gen) * 100)].second;
            
        // Choose a random coefficient for each node
        uniform_real_distribution<float> dis_2(0, config->gen_max);
        float u = dis_2(gen);
        float v = dis_2(gen);
        float w = dis_2(gen);

        // Normalize weights such that their sum equals 1
        float total = u + v + w;
        u /= total;
        v /= total;
        w /= total;
    
        // Obtain a training node in the middle of the 3 selected nodes
        for(int j = 0; j < config->dimensions; j++){
            training[i][j] = u * nodes[i][j] + v * nodes[index_second][j] + w * nodes[index_third][j];
        }
    }
    save_fvecs(config->training_file, training, config->dimensions, config->num_training);
}