#ifndef GRASP_H
#define GRASP_H

#include "../hnsw.h"

// Main algorithms
void learn_edge_importance(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, float** queries, std::ofstream* results_file = nullptr);
void learn_cost_benefit(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, float** training, int num_keep);
void normalize_weights(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, float lambda, float temperature);

// Helper functions
double calculate_weight_change(Config* config, std::vector<std::pair<float, int>>& original_nearest, std::vector<std::pair<float, int>>& sample_nearest, std::ofstream* results_file);
void prune_edges(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, int num_keep);
void sample_subgraph(Config* config, std::vector<Edge*>& edges, float lambda);
void update_weights(Config* config, HNSW* hnsw, float** training, int num_neighbors, std::ofstream* results_file);
float compute_lambda(float final_keep, float initial_keep, int k, int num_iterations, int c);
std::pair<float,float> find_max_min(Config* config, HNSW* hnsw);
float binary_search(Config* config, std::vector<Edge*>& edges, float left, float right, float target, float temperature);
void load_training(Config* config, float** nodes, float** training, int num_training, bool is_generating = false);
void remove_duplicates(Config* config, float** training, float** other, int other_num);

#endif