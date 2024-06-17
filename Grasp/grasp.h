#ifndef GRASP_H
#define GRASP_H

#include "../HNSW/hnsw.h"

//Stage 2 function
void learn_edge_importance(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, float** nodes, float** queries);
float lambda_calculate (float sigma, float lambda_0, int k, int max_K, int c); 
void randome_subgraph(Config* config, HNSW* hnsw); 
void Binomial_weight_Normailization (Config* config, HNSW* hnsw, float lambda, float temprature);

//Normalization Helper functions
int num_of_edges_function (Config* config, HNSW* hnsw);
float find_probability_edge (float weight, float temprature, float mu);
std::pair<float,float> find_max_min  (Config* config, HNSW* hnsw);
float binary_search(Config* config, HNSW* hnsw, float left, float right, float target, float temprature);

//Stage 3 function
void prune_edges(Config* config, HNSW* hnsw, int num_keep);

#endif