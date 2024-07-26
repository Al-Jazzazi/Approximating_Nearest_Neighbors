#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <math.h>
#include <utility>
#include <map> 

using namespace std;

class Config {
public:
    // File Setup
    std::string dataset = "sift";
    int num_return = 1;
    std::string runs_prefix = "./runs/hnsw_sift/gt1_";
    // std::string runs_prefix = "runs/test/testing_finding_neighbors/_";
    std::string loaded_graph_file = "./runs/hnsw_sift/graph_num_return_50.bin";
    bool load_graph_file = true;
    int dimensions = dataset == "sift" ? 128 : dataset == "deep" ? 256 : dataset == "glove" ? 200 : 960;
    int num_nodes = 1000000;
    int num_training = 100000;
    int num_queries = dataset == "sift" || dataset == "glove" ? 10000 : 1000;

    // Interpreted File Setup
    std::string dataset_prefix = "./exports/" + dataset + "/" + dataset;
    std::string load_file = dataset_prefix + "_base.fvecs";
    std::string query_file =  dataset == "sift" ? dataset_prefix + "_query.fvecs" : dataset_prefix + "_learn.fvecs";
    std::string groundtruth_file = num_nodes < 1000000 ? "" : dataset == "sift" ? dataset_prefix + "_groundtruth.ivecs" : dataset_prefix + "_groundtruth_10000.ivecs";
    std::string training_file = dataset_prefix + "_learn.fvecs";
    std::string loaded_info_file = std::regex_replace(std::regex_replace(loaded_graph_file, std::regex("graph"), "info"), std::regex("bin"), "txt");


   
    // HNSW Construction
    const bool use_heuristic = true;
    int max_connections = dataset == "gist" ? 24 : dataset == "glove" ? 16 : 14;
    int max_connections_0 = max_connections;
    int optimal_connections = max_connections;
    double scaling_factor = 1 / log(max_connections);

    // Multiple Entry Points
    const bool single_ep_construction = true;
    const bool single_ep_query = true;
    const bool single_ep_training = true;
    int ef_construction = 500;
    int ef_search = 400;
    int ef_search_upper = 1;
    int k_upper = 1;

    // Termination Parameters
    const bool use_distance_termination = false;
    const bool combined_termination = false; 
    const bool use_number_of_distances = true; 
    const bool use_latest = false;
    const bool use_break = false;
    const bool use_groundtruth_termination = false;  // Use groundtruth to terminate search early
    const bool export_oracle = false;  // Log where the neighbors are found per query
    const bool use_oracle_2 = false;
    std::string oracle_file = std::regex_replace(std::regex_replace(loaded_graph_file, std::regex("graph"), "info"), std::regex("bin"), "txt");
    int number_of_distance_termination_per_q = 200;  // Used if use_number_of_distances = true
    int oracle_termination_total = 10000;  // Used if oracle_file exists
    float termination_alpha = 0.5;  // Used for distance-only termination (not combined)
    float alpha_break = 1.5;
    float efs_break = 1.5;
    const std::map<std::string, std::pair<float, float>> bw = {
        {"deep", {0.197, -300.85}},
        {"sift", {0.2108, -339.64}},
        {"gist", {0.1114, -414.44}},
        {"glove", {0, 0}}
    };
    const std::map<std::string, std::pair<float, float>> alpha = {
        {"50 deep", {0.0185, 0.2273}}, 
        {"10 deep", {0.0169, 0.2548}},
        {"1 deep", {0.0151, 0.2857}},
        {"50 sift", {0.0269f, 0.1754}},
        {"10 sift", {0.0244, 0.2155}},
        {"1 sift", {0.0222, 0.2558}},
        {"50 gist", {0.013f, 0.2454}}, 
        {"10 gist", {0.0111f, 0.2707}},
        {"1 gist", {0.0093, 0.2964}},
        {"50 glove", {0, 0}}, 
        {"10 glove", {0, 0}},
        {"1 glove", {0, 0}}
    };

    // Interpreted Termination Parameters
    string nm = std::to_string(num_return) + " " + dataset;
    float bw_slope = use_break ? bw.at(dataset).first : 0; 
    float bw_intercept = use_break ? bw.at(dataset).second : 0;
    float alpha_coefficient = use_break ? alpha.at(nm).first : 0;
    float alpha_intercept = use_break ? alpha.at(nm).second : 0;

    // HNSW Training
    const bool use_grasp = false;  // Make sure use_grasp and use_cost_benefit are not both on at the same time
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
    int weight_formula = 0;  // 0 = ratio of average distances, 1 = average of distance ratios, 2 = discounted cumulative gain
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
    std::vector<int> benchmark_ef_search = {};
    // std::vector<int> benchmark_ef_search = {200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000};
    std::vector<float> benchmark_termination_alpha = {};
    // std::vector<float> benchmark_termination_alpha = {0.35,0.355,0.36,0.365,0.37,0.375,0.38,0.385,0.39,0.4,0.41,0.42,0.425,0.43,0.44,0.45};
    std::vector<float> benchmark_learning_rate = {};
    std::vector<float> benchmark_initial_temperature = {};
    std::vector<float> benchmark_decay_factor = {};
    std::vector<float> benchmark_initial_keep_ratio = {};
    std::vector<float> benchmark_final_keep_ratio = {};
    std::vector<float> benchmark_stinky_points = {};
    std::vector<int> benchmark_grasp_loops = {};
    std::vector<int> benchmark_grasp_subloops = {};
    // std::vector<int> benchmark_num_of_distance_termination = {2000, 5000,7500, 10000,12500,15000,17500, 20000,22500,25000,27500, 30000,32500,35000};
    std::vector<int> benchmark_num_of_distance_termination = {};
    std::vector<int> benchmark_oracle_termination_total = {};

    // Debugging Flags
    const bool export_benchmark = true;
    const bool export_graph = true;
    const bool export_histograms = false;
    const bool export_weight_updates = true;
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
    const bool debug_insert = false;
    const bool debug_search = false;
    const bool print_graph = false;
    const bool export_queries = false;
    const bool export_indiv = false;
    const bool benchmark_print_neighbors = false;
    const bool benchmark_print_missing = false;
    int debug_query_search_index = -1;

    // Dataset Metrics Parameters
    std::string metrics_file = "./runs/dataset_metrics.txt";
    std::string metrics_dataset1_prefix = "./exports/deep/deep_base";
    std::string metrics_dataset2_prefix = "./exports/deep/deep_query";
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
