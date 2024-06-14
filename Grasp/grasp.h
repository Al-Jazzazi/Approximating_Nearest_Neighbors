#ifndef GRASP_H
#define GRASP_H

#include "../HNSW/hnsw.h"

void learn_edge_importance(Config* config, HNSW* hnsw, float** nodes, float** queries);

#endif