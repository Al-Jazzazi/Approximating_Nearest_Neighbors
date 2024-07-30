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

    for (int i = 0; i< config->num_training; i++){
        // Find top 100 nearest neighbors to each node
        training[i] = new float[config->dimensions];
        vector<Edge*> path;
        pair<int, float*> query = make_pair(i, nodes[i]);
        vector<pair<float, int>> nearest_neighbors = hnsw->nn_search(config, path, query, config->num_return, true); //finds closest 100 neighbors

        // Choose 2 random nearest neighbors
        mt19937 gen(config->graph_seed);
        uniform_int_distribution<> dis(0, 99);
        static std::random_device rd;
        int index_second =  nearest_neighbors[dis(gen)].second;
        int index_third = nearest_neighbors[dis(gen)].second;
            
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