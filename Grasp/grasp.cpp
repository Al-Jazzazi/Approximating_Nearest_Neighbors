#include <iostream>
#include <vector>
#include "grasp.h"
#include "../HNSW/hnsw.h"

using namespace std;

/**
 * Alg 1
 */
void learn_edge_importance(Config* config, HNSW* hnsw, vector<Edge*>& edges, float** nodes, float** queries) {
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

void prune_edges(Config* config, HNSW* hnsw, int num_keep) {
    // Mark lowest weight edges for deletion
    auto compare = [](Edge* lhs, Edge* rhs) { return lhs->weight < rhs->weight; };
    priority_queue<Edge*, vector<Edge*>, decltype(compare)> remaining_edges(compare);
    for (int i = 0; i < hnsw->num_nodes; i++) {
        for (int j = 0; j < hnsw->mappings[i][0].size(); j++) {
            remaining_edges.push(&hnsw->mappings[i][0][j]);
            if (remaining_edges.size() > num_keep) {
                remaining_edges.top()->is_enabled = false;
                remaining_edges.pop();
            }
        }
    }
    // Remove all edges in layer 0 that are marked for deletion
    for (int i = 0; i < hnsw->num_nodes; i++) {
        for (int j = hnsw->mappings[i][0].size() - 1; j >= 0; j--) {
            vector<Edge>& edges = hnsw->mappings[i][0];
            if (!edges[j].is_enabled) {
                edges[j] = layer_edges[edges.size() - 1];
                edges.pop_back();
            }
        }
    }
}