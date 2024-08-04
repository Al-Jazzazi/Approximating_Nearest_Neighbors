#include <iostream>
#include <vector>
#include "hnsw.h"
#include "grasp.h"

int main() {
    // Load config
    Config* config = new Config();
    config->num_return = 100;

    // Load nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** training = new float*[config->num_training];
    load_training(config, nodes, training, config->num_training);
    float** generated = nullptr; 
    generated = new float*[config->num_training_generated];

    // Create HNSW graph using training set
    HNSW* hnsw = NULL;
    hnsw = new HNSW(config, nodes);
    if (config->load_graph_file) {
        hnsw->from_files(config, true);
    } else {
        for (int i = 1; i < config->num_nodes; ++i) {
            hnsw->insert(config, i);
        }
    }

    mt19937 gen(config->graph_seed);
    uniform_real_distribution<float> dis(0, 0.9999999);
    for (int i = 0; i< config->num_training_generated; i++){
        // Choose a random node out of the source dataset
        generated[i] = new float[config->dimensions];
        int index_first = dis(gen) * config->num_training;
        pair<int, float*> query = make_pair(index_first, training[index_first]);

        // Choose 2 random nearest neighbors out of top 100
        vector<Edge*> path;
        vector<pair<float, int>> nearest_neighbors = hnsw->nn_search(config, path, query, config->num_return, true);
        int index_second =  nearest_neighbors[static_cast<int>(dis(gen) * 100)].second;
        int index_third = nearest_neighbors[static_cast<int>(dis(gen) * 100)].second;
            
        // Choose a random coefficient for each node
        float u = dis(gen);
        float v = dis(gen);
        float w = dis(gen);

        // Normalize coefficients such that they add up to 1
        float total = u + v + w;
        u /= total;
        v /= total;
        w /= total;
    
        // Generate a node in the middle of the 3 selected nodes
        for(int j = 0; j < config->dimensions; j++){
            generated[i][j] = u * training[index_first][j] + v * nodes[index_second][j] + w * nodes[index_third][j];
        }
    }
    save_fvecs(config->generated_training_file, generated, config->num_training_generated, config->dimensions);
}