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

extern std::ofstream* debug_file;

class Edge {
public:
    // HNSW
    int target;
    float distance;

    // GraSP
    Edge* prev_edge;
    float weight; 
    float stinky;
    bool ignore;
    float probability_edge;
    unsigned int num_of_updates; 
    

    // Cost-Benefit
    int benefit;
    int cost;

    Edge();
    Edge(int target, float distance, int initial_cost = 0, int initial_benefit = 0);
};

class HNSW {
    friend std::ostream& operator<<(std::ostream& os, const HNSW& hnsw);
public:
    // Stores nodes by node index, then dimensions
    float** nodes;
    // Stores edges in adjacency list by node index, then layer number, then connection pair
    std::vector<std::vector<std::vector<Edge>>> mappings;
    int entry_point;
    int num_layers;
    int num_nodes;
    int num_dimensions;

    // Probability function
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    double normal_factor;

    // Statistics
    long long int layer0_dist_comps;
    long long int layer0_dist_comps_per_q; 
    long long int upper_dist_comps;
    long long int actual_beam_width;
    long long int processed_neighbors;
    long long int total_neighbors;
    long long int num_distance_termination; 
    long long int num_original_termination;
    long long int total_path_size;
    std::vector<float> percent_neighbors;
    std::vector<int> cur_groundtruth;


    HNSW(Config* config, float** nodes);
    void reset_statistics();
    void search_queries(Config* config, float** queries);
    std::vector<Edge*> get_layer_edges(Config* config, int layer);
    void find_direct_path(std::vector<Edge*>& path, std::vector<std::pair<float, int>>& entry_points);
    bool should_terminate(Config* config, std::priority_queue<std::pair<float, int>>& top_k, std::pair<float, int>& top_1, float close_squared, float far_squared, float far_extension_sqaured, bool is_querying, int layer_num);
    float calculate_average_clustering_coefficient();

    // Main algorithms
    void insert(Config* config, int query);
    void search_layer(Config* config, float* query, std::vector<Edge*>& path, std::vector<std::pair<float, int>>& entry_points, int num_to_return, int layer_num, bool is_querying = false, bool is_training = false, bool is_ignoring = false);
    void select_neighbors_heuristic(Config* config, float* query, std::vector<Edge>& candidates, int num_to_return, int layer_num, bool extend_candidates = false, bool keep_pruned = true);
    std::vector<std::pair<float, int>> nn_search(Config* config, std::vector<Edge*>& path, std::pair<int, float*>& query, int num_to_return, bool is_querying = true, bool is_training = false, bool is_ignoring = false);
};

// Helper functions
HNSW* init_hnsw(Config* config, float** nodes);
float calculate_l2_sq(HNSW* hnsw, int layer, float* a, float* b, int size);
float calculate_l2_sq(float* a, float* b, int size);
void load_fvecs(const std::string& file, const std::string& type, float** nodes, int num, int dim, bool has_groundtruth);
void load_ivecs(const std::string& file, std::vector<std::vector<int>>& results, int num, int dim);
void save_ivecs(const string& file, vector<vector<int>>& results);
void load_hnsw_files(Config* config, HNSW* hnsw, float** nodes, bool is_benchmarking = false);
void load_hnsw_graph(Config* config, HNSW* hnsw, std::ifstream& graph_file, float** nodes, int num_nodes, int num_layers);
void save_hnsw_files(Config* config, HNSW* hnsw, const std::string& name, long int duration);
void load_nodes(Config* config, float** nodes);
void load_queries(Config* config, float** nodes, float** queries);
void load_oracle(Config* config, std::vector<std::pair<int, int>>& result);
void knn_search(Config* config, std::vector<std::vector<int>>& actual_neighbors, float** nodes, float** queries);

#endif