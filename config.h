#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <string>
#include <vector>

class Config {
public:
    // Datasets
    std::string load_file = "./exports/sift/sift_base.fvecs";
    std::string query_file = "./exports/sift/sift_query.fvecs";
    std::string groundtruth_file = "";
    std::string training_file = "";
    int dimensions = 128;
    int num_nodes = 10000;
    int num_training = 1000;
    int num_queries = 1000;

    // Save/Load Files
    std::string runs_dir = "./runs/";
    std::string benchmark_file = "./runs/grasp_benchmark.txt";
    std::string histogram_prefix = "./runs/histogram";
    std::string save_file_prefix = "./runs/grasp";
    std::string hnsw_graph_file = "./runs/grasp_graph_num_return_50.bin";
    std::string hnsw_info_file = "./runs/grasp_info_num_return_50.txt";
    bool load_graph_file = false;
    bool export_benchmark = true;
    bool export_graph = false;

    // HNSW Construction
    int optimal_connections = 14;
    int max_connections = 14;
    int max_connections_0 = 14;
    int ef_construction = 500;
    double scaling_factor = 0.379;
    // HNSW Search
    int ef_search = 400;
    int num_return = 50;
    // Enforces a single entry point for graph construction. Searching will always be single entry point
    bool single_entry_point = true;
    bool use_heuristic = true;
    bool use_grasp = true;
    
    // GraSP Training
    float learning_rate = 0.1;
    float initial_temperature = 1;
    float decay_factor = 0.8;
    float initial_keep_ratio = 0.9;
    float final_keep_ratio = 0.7;
    int keep_exponent = 3;
    int grasp_loops = 20;
    int grasp_subloops = 1;
    // -1 = use num_return instead of num_return_training
    int num_return_training = -1;
    // 0 = all edges on original path, 1 = only ignored edges, 2 = exclude edges on sample path
    int weight_selection_method = 0;
    bool print_weight_updates = true;
    bool export_weight_updates = true;
    bool use_dynamic_sampling = false;
    bool use_stinky_points = false; 
    float stinkyValue = 0.00005;
    int interval_for_weight_histogram = 1; 
    int interval_for_num_of_updates_histogram = 10; 

    // Benchmark parameters
    std::vector<int> benchmark_num_return = {1, 10, 50};
    std::vector<int> benchmark_optimal_connections = {};
    std::vector<int> benchmark_max_connections = {};
    std::vector<int> benchmark_max_connections_0 = {};
    std::vector<int> benchmark_ef_construction = {};
    std::vector<int> benchmark_ef_search = {};
    std::vector<float> benchmark_learning_rate = {};
    std::vector<float> benchmark_initial_temperature = {};
    std::vector<float> benchmark_decay_factor = {};
    std::vector<float> benchmark_initial_keep_ratio = {};
    std::vector<float> benchmark_final_keep_ratio = {};
    std::vector<float> benchmark_stinky_points = {};
    std::vector<int> benchmark_grasp_loops = {};
    std::vector<int> benchmark_grasp_subloops = {};
    bool benchmark_print_neighbors = false;
    bool benchmark_print_missing = false;

    // Generation Settings
    int graph_seed = 0;
    int query_seed = 100000;
    int training_seed = 100000;
    int insertion_seed = 1000000;
    int gen_min = 0;
    int gen_max = 100000;
    int gen_decimals = 2;

    // HNSW Flags
    // Note: Distance ties will make the found percentage lower
    bool run_search = true;
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
    bool export_queries = false;
    bool export_indiv = true;
    int debug_query_search_index = -1;

    // HNSW Save Parameters
    std::vector<int> save_optimal_connections = {7, 14, 25};
    std::vector<int> save_max_connections = {11, 18, 30};
    std::vector<int> save_max_connections_0 = {14, 28, 50};
    std::vector<int> save_ef_constructions = {21, 42, 75};
    int num_graphs_saved = 3;

    // Dataset Metrics Parameters
    std::string metrics_file = "./runs/dataset_metrics.txt";
    std::string metrics_dataset1_prefix = "./exports/sift/sift_learn";
    std::string metrics_dataset2_prefix = "./exports/sift/sift_query";
    bool compare_datasets = true;
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