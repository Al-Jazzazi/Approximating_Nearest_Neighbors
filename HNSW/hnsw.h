#ifndef HNSW_H
#define HNSW_H

#include <vector>
#include <map>
#include <fstream>
#include <queue>
#include <random>
#include <functional>

extern long long int layer0_dist_comps;
extern long long int upper_dist_comps;

extern std::ofstream* debug_file;

class Config {
public:
    const std::string load_file = "./exports/sift/sift_base.fvecs";
    const std::string query_file = "./exports/sift/sift_query.fvecs";
    const std::string groundtruth_file = "";
    const std::string export_dir = "runs/";

    int graph_seed = 0;
    int query_seed = 100000;
    int insertion_seed = 1000000;

    int gen_min = 0;
    int gen_max = 100000;
    int gen_decimals = 2;
    float reinsert_percent = 0.0;
    
    // Enforces a single entry point for graph construction. Searching will always be single entry point
    bool single_entry_point = true;

    int dimensions = 128;
    int num_nodes = 10000;
    int optimal_connections = 10;
    int max_connections = 30;
    int max_connections_0 = 30;
    int ef_construction = 50;
    double scaling_factor = 0.368;

    int ef_search = 300;
    int num_queries = 10000;
    int num_return = 20;

    // Note: Distance ties will make the found percentage lower
    bool print_results = false;
    bool print_actual = false;
    bool print_indiv_found = false;
    bool print_total_found = true;

    // Log where the neighbors are found per query
    bool gt_dist_log = false;
    // Use groundtruth to terminate search early
    bool gt_smart_termination = true;

    bool debug_insert = false;
    bool debug_search = false;
    bool print_graph = false;

    bool export_graph = true;
    bool export_queries = true;
    bool export_indiv = true;

    int debug_query_search_index = -1;
};

class Edge {
public:
    int target;
    float distance;
    float weight; 

    Edge();
    Edge(int target, float distance, float weight = 0.5);
    bool operator>(const Edge& rhs) const;
    bool operator<(const Edge& rhs) const;
};

class HNSW {
public:
    int node_size;
    float** nodes;
    // This vector stores vectors by node index, then layer number, then connection pair
    std::vector<std::vector<std::vector<Edge>>> mappings;
    int entry_point;
    int layers;

    HNSW(int node_size, float** nodes);
};

// Helper functions
float calculate_l2_sq(float* a, float* b, int size, int layer);
void load_fvecs(const std::string& file, const std::string& type, float** nodes, int num, int dim, bool has_groundtruth);
void load_ivecs(const std::string& file, std::vector<std::vector<int>>& results, int num, int dim);

// Loading nodes and graph
void load_hnsw_file(Config* config, HNSW* hnsw, float** nodes, const std::string& graph_file_name, const std::string& info_file_name, bool is_benchmarking = false);
void load_hnsw_graph(HNSW* hnsw, std::ifstream& graph_file, float** nodes, int num_nodes, int num_layers);
void load_nodes(Config* config, float** nodes);
void load_queries(Config* config, float** nodes, float** queries);

// Main algorithms
HNSW* insert(Config* config, HNSW* hnsw, int query, int est_con, int max_con, int ef_con, float normal_factor, std::function<double()> rand);
void search_layer(Config* config, HNSW* hnsw, float* query, std::vector<std::pair<float, int>>& entry_points, int num_to_return, int layer_num);
std::vector<std::pair<float, int>> nn_search(Config* config, HNSW* hnsw, std::pair<int, float*>& query, int num_to_return, int ef_con, std::vector<int>& path);

// Executing HNSW
bool sanity_checks(Config* config);
HNSW* init_hnsw(Config* config, float** nodes);
void insert_nodes(Config* config, HNSW* hnsw);
void print_hnsw(Config* config, HNSW* hnsw);
void run_query_search(Config* config, HNSW* hnsw, float** queries);
void export_graph(Config* config, HNSW* hnsw, float** nodes);
void reinsert_nodes(Config* config, HNSW* hnsw);
void delete_node(Config* config, HNSW* hnsw, int index);
std::vector<int> get_layer(Config* config, HNSW* hnsw, int layer);

#endif