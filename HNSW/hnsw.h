#ifndef HNSW_H
#define HNSW_H

#include <vector>
#include <map>
#include <fstream>
#include <queue>
#include <random>
#include <functional>
#include <immintrin.h>
#include "../config.h"

extern long long int layer0_dist_comps;
extern long long int upper_dist_comps;

extern std::ofstream* debug_file;

class Edge {
public:
    int target;
    float distance;
    float weight; 
    bool ignore;
    float probability_edge;

    Edge();
    Edge(int target, float distance, float weight = 50, bool ignore = false, float probability_edge = 0.5);
    bool operator>(const Edge& rhs) const;
    bool operator<(const Edge& rhs) const;
};

class HNSW {
    friend std::ostream& operator<<(std::ostream& os, const HNSW& hnsw);
public:
    // This stores nodes by node index, then dimensions
    float** nodes;
    // This stores edges in an adjacency list by node index, then layer number, then connection pair
    std::vector<std::vector<std::vector<Edge>>> mappings;
    int entry_point;
    int num_layers;
    int num_nodes;
    int num_dimensions;
    
    // Probability function
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    double normal_factor;

    HNSW(Config* config, float** nodes);
    void export_graph(Config* config);
    void search_queries(Config* config, float** queries);
    std::vector<Edge*> get_layer_edges(Config* config, int layer);
    
    // Main algorithms
    void insert(Config* config, int query);
    void search_layer(Config* config, float* query, std::vector<std::vector<Edge*>>& path, std::vector<std::pair<float, int>>& entry_points, int num_to_return, int layer_num, bool is_ignoring = false, bool add_stinky = false);
    void select_neighbors_heuristic(Config* config, float* query, std::vector<Edge>& candidates, int num_to_return, int layer_num, bool extend_candidates = false, bool keep_pruned = true);
    std::vector<std::pair<float, int>> nn_search(Config* config, std::vector<std::vector<Edge*>>& path, std::pair<int, float*>& query, int num_to_return, bool is_ignoring = false, bool add_stinky = false);
};

// Helper functions
HNSW* init_hnsw(Config* config, float** nodes);
float calculate_l2_sq(float* a, float* b, int size, int layer);
void load_fvecs(const std::string& file, const std::string& type, float** nodes, int num, int dim, bool has_groundtruth);
void load_ivecs(const std::string& file, std::vector<std::vector<int>>& results, int num, int dim);
void load_hnsw_file(Config* config, HNSW* hnsw, float** nodes, bool is_benchmarking = false);
void load_hnsw_graph(HNSW* hnsw, std::ifstream& graph_file, float** nodes, int num_nodes, int num_layers);
void load_nodes(Config* config, float** nodes);
void load_queries(Config* config, float** nodes, float** queries);

#endif