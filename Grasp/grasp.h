#ifndef GRASP_H
#define GRASP_H

#include "../HNSW/hnsw.h"

void learn_edge_importance(Config* config, HNSW* hnsw, std::vector<Edge*>& edges, float** nodes, float** queries);
void prune_edges(Config* config, HNSW* hnsw, int num_keep);

#endif