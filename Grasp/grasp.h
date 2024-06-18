#ifndef GRASP_H
#define GRASP_H

#include "../HNSW/hnsw.h"

// Stage 2 functions
void learn_edge_importance(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, float** nodes, float** queries);
void sample_subgraph(Config* config, HNSW* hnsw, float lambda); 
void normalize_weights(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, float lambda, float temperature);
float compute_lambda(float final_keep, float initial_keep, int k, int num_iterations, int c);

// Normalization Helper functions
std::pair<float,float> find_max_min(Config* config, HNSW* hnsw);
float binary_search(Config* config, std::vector<Edge*>& edges, float left, float right, float target, float temperature);

// Stage 3 functions
void prune_edges(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, int num_keep);

#endif