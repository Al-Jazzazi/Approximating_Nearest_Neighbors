#include <iostream>
#include <vector>
#include "grasp.h"
#include "../HNSW/hnsw.h"

using namespace std;

/**
 * Alg 1
 */
void learn_edge_importance(Config* config, HNSW* hnsw, float** nodes, float** queries) {
    float temperature = config->initial_temperature;
    for (int k = 0; k < config->grasp_iterations; k++) {
        // TODO: formulas

        for (int i = 0; i < config->num_training; i++) {
            // Find the nearest neighbor using both the original and sampled graphs
            pair<int, float*> query = make_pair(i, queries[i]);
            vector<vector<Edge*>> sample_path;
            vector<vector<Edge*>> original_path;
            vector<pair<float, int>> sample_nearest = hnsw->nn_search(config, sample_path, query, 1);
            vector<pair<float, int>> original_nearest = hnsw->nn_search(config, original_path, query, 1);
            
            // If the nearest neighbor differs, increase the weight importances
            if (original_nearest[0].second != sample_nearest[0].second) {
                for (int j = 0; j < original_path[0].size(); j++) {
                    float sample_distance = calculate_l2_sq(nodes[sample_nearest[0].second], queries[i], config->dimensions, 0);
                    float original_distance = calculate_l2_sq(nodes[original_nearest[0].second], queries[i], config->dimensions, 0);
                    float& weight = original_path[0][j]->weight;
                    if (original_distance != 0) {
                        original_path[0][j]->weight = original_path[0][j]->weight + (sample_distance / original_distance - 1) * config->learning_rate;
                    }
                }
            }
        }
    }
}