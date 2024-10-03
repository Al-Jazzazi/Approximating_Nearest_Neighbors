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
#include "utils.h"

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

    //Overload Operators
    bool operator==(const Edge& other) const {
        return this->distance == other.distance;
    }

    bool operator!=(const Edge& other) const {
        return !(*this == other);
    }

    bool operator<(const Edge& other) const {
        return this->distance < other.distance;
    }

    bool operator>(const Edge& other) const {
        return this->distance > other.distance;
    }

    bool operator<=(const Edge& other) const {
        return !(*this > other);
    }

    bool operator>=(const Edge& other) const {
        return !(*this < other);
    }



    Edge();
    Edge(int target, float distance, int initial_cost = 0, int initial_benefit = 0);
};

class HNSW {
    friend std::ostream& operator<<(std::ostream& os, const HNSW& hnsw);
public:
    float** nodes; // Node index, then dimensions
    std::vector<std::vector<std::vector<Edge>>> mappings; // Node index, then layer number, then neighbors
    int entry_point;
    int num_layers;
    int num_nodes;
    int num_dimensions;

    // Probability function
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    double normal_factor;

    // Statistics
    int layer0_dist_comps_per_q; 
    long long int layer0_dist_comps;
    long long int upper_dist_comps;
    long long int processed_neighbors;
    long long int total_neighbors;
    long long int num_distance_termination; 
    long long int num_original_termination;
    long long int total_path_size;
    long long int candidates_popped;
    long long int candidates_size;
    long long int candidates_without_if;
    std::vector<float> percent_neighbors;
    std::vector<int> cur_groundtruth;

    static std::map<int,std::vector<int>> candidate_popping_times;


    HNSW(Config* config, float** nodes);
    void to_files(Config* config, const std::string& graph_name, long int construction_duration = 0);
    void from_files(Config* config, bool is_benchmarking = false);
    void reset_statistics();
    std::vector<Edge*> get_layer_edges(Config* config, int layer);
    void find_direct_path(std::vector<Edge*>& path, std::vector<std::pair<float, int>>& entry_points);
    bool should_terminate(Config* config, std::priority_queue<std::pair<float, int>>& top_k, std::pair<float, int>& top_1, float close_squared, float far_squared, bool is_querying, int layer_num, int candidates_popped_per_q);
    float calculate_average_clustering_coefficient();
    float calculate_global_clustering_coefficient();
    float calculate_distance(float* a, float* b, int size, int layer);
    void  calculate_termination(Config* config);

    // Main algorithms
    void insert(Config* config, int query);
    void search_layer(Config* config, float* query, std::vector<Edge*>& path, std::vector<std::pair<float, int>>& entry_points, int num_to_return, int layer_num, bool is_querying = false, bool is_training = false, bool is_ignoring = false, int* total_cost = nullptr);
    void select_neighbors_heuristic(Config* config, float* query, std::vector<Edge>& candidates, int num_to_return, int layer_num, bool extend_candidates = false, bool keep_pruned = true);
    std::vector<std::pair<float, int>> nn_search(Config* config, std::vector<Edge*>& path, std::pair<int, float*>& query, int num_to_return, bool is_querying = true, bool is_training = false, bool is_ignoring = false, int* total_cost = nullptr);
    void search_queries(Config* config, float** queries);
};



#endif