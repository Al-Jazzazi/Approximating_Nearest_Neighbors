#ifndef GRASP_H
#define GRASP_H

#include "../HNSW/hnsw.h"

//Stage 2 function
void learn_edge_importance(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, float** nodes, float** queries);
void sample_subgraph(Config* config, HNSW* hnsw); 
void normalize_weights(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, float lambda, float temperature);

//Normalization Helper functions
std::pair<float,float> find_max_min(Config* config, HNSW* hnsw);
float binary_search(Config* config, HNSW* hnsw, float left, float right, float target, float temperature);

//Stage 3 function
void prune_edges(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, int num_keep);

#endif