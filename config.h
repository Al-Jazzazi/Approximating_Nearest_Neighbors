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
    std::string export_dir = "./HNSW/runs/";
    std::string hnsw_graph_file = "./HNSW/runs/random_graph_graph_0.bin";
    std::string hnsw_info_file = "./HNSW/runs/random_graph_info_0.txt";
    bool load_graph_file = false;

    // HNSW Construction
    int dimensions = 128;
    int num_nodes = 10000;
    int optimal_connections = 7;
    int max_connections = 11;
    int max_connections_0 = 14;
    int ef_construction = 21;
    double scaling_factor = 0.368;
    // Enforces a single entry point for graph construction. Searching will always be single entry point
    bool single_entry_point = true;

    // HNSW Search
    int ef_search = 300;
    int num_queries = 1000;
    int num_return = 50;

    // GraSP Training
    float learning_rate = 0.1;
    float initial_temperature = 1;
    float decay_factor = 0.8;
    float initial_keep_ratio = 0.95;
    float final_keep_ratio = 0.6;
    int keep_exponent = 1;
    int grasp_iterations = 20;
    int num_training = 1000;

    // Generation Settings
    int graph_seed = 0;
    int query_seed = 100000;
    int insertion_seed = 1000000;
    int gen_min = 0;
    int gen_max = 100000;
    int gen_decimals = 2;

    // HNSW/hnsw.cpp parameters
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
    bool export_graph = true;
    bool export_queries = true;
    bool export_indiv = true;
    int debug_query_search_index = -1;

    // HNSW/benchmark.cpp parameters
    std::vector<int> benchmark_optimal_connections = {3, 7, 10, 20, 30};
    std::vector<int> benchmark_max_connections = {10, 20, 30, 40, 50};
    std::vector<int> benchmark_max_connections_0 = {10, 20, 30, 40, 50};
    std::vector<int> benchmark_ef_construction = {10, 30, 50, 70, 90};
    std::vector<int> benchmark_ef_search = {100, 300, 500, 700, 1000};
    std::vector<int> benchmark_num_return = {10, 50, 100, 150, 200};
    std::string benchmark_file = "./HNSW/runs/random_graph_results.txt";
    bool export_benchmark = true;
    bool benchmark_print_neighbors = false;
    bool benchmark_print_missing = false;

    // HNSW/save_hnsw.cpp parameters
    std::vector<int> save_optimal_connections = {7, 14, 25};
    std::vector<int> save_max_connections = {11, 18, 30};
    std::vector<int> save_max_connections_0 = {14, 28, 50};
    std::vector<int> save_ef_constructions = {21, 42, 75};
    std::string save_file_prefix = "./HNSW/runs/random_graph";
    int num_graphs_saved = 3;

    // HNSW/dataset_metrics parameters
    std::string metrics_file = "runs/dataset_metrics.txt";
    std::string metrics_dataset1_prefix = "./exports/sift/sift_base";
    std::string metrics_dataset2_prefix = "./exports/sift/sift_query";
    bool compare_datasets = true;
    int comparison_num_nodes = 1000;
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