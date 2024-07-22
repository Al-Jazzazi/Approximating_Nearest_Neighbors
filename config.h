#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <math.h>

class Config {
public:
    // File Setup
    std::string dataset_prefix = "./exports/sift/sift";
    std::string runs_prefix = "./runs/";
    std::string loaded_graph_file = "./runs/hnsw_sift/graph_num_return_50.bin";
    bool load_graph_file = true;
    int dimensions = 128;
    int num_nodes = 1000000;
    int num_training = 100000;
    int num_queries = 10000;
    int num_return = 50;

    // Interpreted File Setup
    std::string load_file = dataset_prefix + "_base.fvecs";
    std::string query_file = dataset_prefix + "_query.fvecs";
    std::string groundtruth_file = num_nodes < 1000000 ? "" : dataset_prefix + "_groundtruth.ivecs";
    std::string training_file = dataset_prefix + "_learn.fvecs";
    std::string loaded_info_file = std::regex_replace(std::regex_replace(loaded_graph_file, std::regex("graph"), "info"), std::regex("bin"), "txt");

    // HNSW Construction
    const bool use_heuristic = true;
    int max_connections = 14;
    int max_connections_0 = max_connections;
    int optimal_connections = max_connections;
    double scaling_factor = 1 / log(max_connections);

    // Beam Search
    const bool single_ep_construction = true;
    const bool single_ep_query = true;
    const bool single_ep_training = true;
    int ef_construction = 500;
    int ef_search = 400;
    int ef_search_upper = 1;
    int k_upper = 1;

    // Termination Parameters
    const bool use_distance_termination = false;
    const bool combined_termination = true; 
    const bool use_latest = true;
    const bool use_break = true;
    float termination_alpha = 0.5;  // Used for distance-only termination (not combined)
    float alpha_break = 1.5;
    float efs_break = 1.5;
    float bw_slope = 0.2108; 
    float bw_intercept = -389.13;
    float alpha_coefficient = 0.0257;
    float alpha_intercept = 0.179;

    // HNSW Training
    const bool use_grasp = true;  // Make sure use_grasp and use_cost_benefit are not both on at the same time
    const bool use_cost_benefit = false;
    const bool use_direct_path = false;
    const bool use_dynamic_sampling = false;
    const bool use_stinky_points = false;
    float stinky_value = 0.00005;
    float learning_rate = 0.1;
    float initial_temperature = 1;
    float decay_factor = 0.8;
    int keep_exponent = 3;
    int grasp_loops = 20;
    int grasp_subloops = 1;
    int weight_selection_method = 0;  // 0 = all edges on original path, 1 = only ignored edges, 2 = exclude edges on sample path
    int weight_formula = 0;  // 0 = original distance formula, 1 = position-based formula
    float initial_keep_ratio = 0.9;
    float final_keep_ratio = 0.7;
    int initial_cost = 1;
    int initial_benefit = 1;
    
    // Grid parameters: repeat all benchmarks for each set of grid values
    std::vector<int> grid_num_return = {}; 
    std::vector<std::string> grid_runs_prefix = {};
    std::vector<std::string> grid_graph_file = {};
    
    // Benchmark parameters
    std::vector<int> benchmark_num_return = {};
    std::vector<int> benchmark_optimal_connections = {};
    std::vector<int> benchmark_max_connections = {};
    std::vector<int> benchmark_max_connections_0 = {};
    std::vector<int> benchmark_ef_construction = {};
    //std::vector<int> benchmark_ef_search = {};
    std::vector<int> benchmark_ef_search = {200, 300, 400, 500, 600, 700, 800 }; //, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000};
    std::vector<float> benchmark_termination_alpha = {};
    // std::vector<float> benchmark_termination_alpha = {0.5, 0.75, 1, 1.25, 1.5, 1.75, 2};
    std::vector<float> benchmark_learning_rate = {};
    std::vector<float> benchmark_initial_temperature = {};
    std::vector<float> benchmark_decay_factor = {};
    std::vector<float> benchmark_initial_keep_ratio = {};
    std::vector<float> benchmark_final_keep_ratio = {};
    std::vector<float> benchmark_stinky_points = {};
    std::vector<int> benchmark_grasp_loops = {};
    std::vector<int> benchmark_grasp_subloops = {};

    // Debugging Flags
    const bool export_benchmark = true;
    const bool export_graph = true;
    const bool export_histograms = true;
    const bool export_weight_updates = false;
    const bool export_training_queries = false; 
    const bool export_negative_values = false; 
    const bool print_weight_updates = true;
    const bool print_neighbor_percent = false;
    const bool print_path_size = false;
    int interval_for_neighbor_percent = 100;
    int interval_for_weight_histogram = 1; 
    int interval_for_num_of_updates_histogram = 1;
    int interval_for_cost_histogram = 10; 
    int interval_for_benefit_histogram = 1; 

    // Generation Settings
    std::string training_set = "";
    const bool generate_our_training = false;
    const bool regenerate_each_iteration = false;
    int num_training_generated = 0;  // 0 = don't generate training queries
    int graph_seed = 0;
    int shuffle_seed = 1;
    int sample_seed = 2;
    int query_seed = 100000;
    int training_seed = 100000;
    int insertion_seed = 1000000;
    int gen_min = 0;
    int gen_max = 100000;
    int gen_decimals = 2;
 
    // Old HNSW Flags
    const bool run_search = true;
    const bool print_results = false;
    const bool print_actual = false;
    const bool print_indiv_found = false;
    const bool print_total_found = false;
    const bool gt_dist_log = false;  // Log where the neighbors are found per query
    const bool gt_smart_termination = true;  // Use groundtruth to terminate search early
    const bool debug_insert = false;
    const bool debug_search = false;
    const bool print_graph = false;
    const bool export_queries = false;
    const bool export_indiv = false;
    const bool benchmark_print_neighbors = false;
    const bool benchmark_print_missing = false;
    int debug_query_search_index = -1;

    // HNSW Save Parameters
    std::vector<int> save_optimal_connections = {7, 14, 25};
    std::vector<int> save_max_connections = {11, 18, 30};
    std::vector<int> save_max_connections_0 = {14, 28, 50};
    std::vector<int> save_ef_constructions = {21, 42, 75};
    int num_graphs_saved = 3;

    // Dataset Metrics Parameters
    std::string metrics_file = "./runs/dataset_metrics.txt";
    std::string metrics_dataset1_prefix = "./exports/deep1M/deep1M_base";
    std::string metrics_dataset2_prefix = "./exports/deep1M/deep1M_query";
    bool compare_datasets = false;
    int comparison_num_nodes = 10000;
    int hopkins_sample_size = 1000;
    int cluster_k = 400;
    int cluster_iterations = 20;

    Config() {
        if (!sanity_checks()) {
            exit(1);
        }
    }

    bool sanity_checks() {
        if (optimal_connections > max_connections) {
            std::cout << "Optimal connections cannot be greater than max connections" << std::endl;
            return false;
        }
        if (optimal_connections > ef_construction) {
            std::cout << "Optimal connections cannot be greater than beam width" << std::endl;
            return false;
        }
        if (num_return > num_nodes) {
            std::cout << "Number of nodes to return cannot be greater than number of nodes" << std::endl;
            return false;
        }
        if (ef_construction > num_nodes) {
            ef_construction = num_nodes;
            std::cout << "Warning: Beam width was set to " << num_nodes << std::endl;
        }
        if (num_return > ef_search) {
            num_return = ef_search;
            std::cout << "Warning: Number of queries to return was set to " << ef_search << std::endl;
        }
        return true;
    }
};

#endif
