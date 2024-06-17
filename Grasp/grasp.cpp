#include <iostream>
#include <vector>
#include "grasp.h"
#include <math.h>
#include <utility>
#include <random>
#include <cfloat> 
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
                remaining_edges.top()->ignore = false;
                remaining_edges.pop();
            }
        }
    }
    // Remove all edges in layer 0 that are marked for deletion
    for (int i = 0; i < hnsw->num_nodes; i++) {
        for (int j = hnsw->mappings[i][0].size() - 1; j >= 0; j--) {
            vector<Edge>& edges = hnsw->mappings[i][0];
            if (!edges[j].ignore) {
                edges[j] = edges[edges.size() - 1];
                edges.pop_back();
            }
        }
    }
}




float lambda_calculate (float sigma, float lambda_0, int k, int max_K, int c){
    return sigma + (lambda_0 - sigma)* pow((1- (float)k/max_K), c);
}


void Binomial_weight_Normailization (Config* config, HNSW* hnsw, float lambda, float temprature){

    int num_of_edges = num_of_edges_function (config, hnsw );
    float target = lambda * num_of_edges;
    pair<float,float> max_min = find_max_min(config, hnsw);
    float avg_w = temprature * log( lambda / (1-lambda));

    float search_range_min = avg_w - max_min.first;
    float search_range_max = avg_w - max_min.second;

    float mu = search(config, hnsw, search_range_min, search_range_max, target);

    for(int i = 0; i < config->num_nodes ; i++){
        for(int k = 0; k< hnsw->mappings[i][0].size(); k++){
            hnsw->mappings[i][0][k].weight += mu;
            hnsw->mappings[i][0][k].probability_edge = find_probability_edge(hnsw->mappings[i][0][k].weight, temprature);
        }
    }

}

int num_of_edges_function (Config* config, HNSW* hnsw){
 int size = 0;
 for(int i = 0; i < config->num_nodes ; i++){
        size += hnsw->mappings[i][0].size();
}
 return size; 
}

pair<float,float> find_max_min  (Config* config, HNSW* hnsw){
    float max_w = 0.0f; 
    float min_w = FLT_MAX; 
    pair<float,float> max_min;
    for(int i = 0; i < config->num_nodes ; i++){
        for(int k = 0; k< hnsw->mappings[i][0].size(); k++){
            if(max_w < hnsw->mappings[i][0][k].weight)
                max_w = hnsw->mappings[i][0][k].weight;
            if(min_w > hnsw->mappings[i][0][k].weight)
                min_w = hnsw->mappings[i][0][k].weight;
        }
    }
    max_min = make_pair(max_w, min_w);
  return max_min;
}

float find_probability_edge (float weight, float temprature){
    return 1/(1+exp(-weight/temprature));
}

float search(Config* config, HNSW* hnsw, float left, float right, float target){
     for(int i = 0; i < config->num_nodes ; i++){
        for(int k = 0; k< hnsw->mappings[i][0].size(); k++){
            ////
        }
    }

}


void randome_subgraph(Config* config, HNSW* hnsw) {
    mt19937 gen(config->graph_seed);
    uniform_real_distribution<float> dis(0, 1);
     for(int i = 0; i < config->num_nodes ; i++){
        for(int k = 0; k< hnsw->mappings[i][0].size(); k++){
            if(hnsw->mappings[i][0][k].probability_edge < dis(gen) )
                 hnsw->mappings[i][0][k].ignore = true; 
            else 
                hnsw->mappings[i][0][k].ignore = false; 
        }
    }

}






