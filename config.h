#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <math.h>
#include <utility>
#include <map> 

class Config {
public:

    // File Setup
    std::string dataset = "sift";
    int num_return = 1;
    int alpha_termination_selection = 3; 
    // alpha * (2 * d_k + d_1)  --> 0 
    // alpha * 2 * d_k + d_1  --> 1 
    // alpha * (d_k + d_1)  + d_k --> 2 
    // alpha * d_1   + d_k   --> 3 
    // alpha * d_k   + d_k  --> 4 
    std::string runs_prefix =  "./runs_fall_2024/run/"+ dataset+"/distance_termination/k="+std::to_string(num_return)+"_distance_termination_" + std::to_string(alpha_termination_selection);

    //  "./runs_fall_2024/vamana/"+ dataset + "/k=" +std::to_string(num_return) + "_" ;

    std::string metric_prefix = "./runs_fall_2024/data_metrics/"+ dataset+"/k=1__full_";
    std::string loaded_graph_file = "./grphs/"+ dataset+"/graph_hnsw_heuristic.bin";

    //  "./grphs/vamana/_graph_vamana_1M_sift.bin";
 
    bool load_graph_file = true;
    int dimensions = dataset == "sift" ? 128 : dataset == "deep" ? 256 : dataset == "deep96" ? 96 : dataset == "glove" ? 200 : 960;
    int num_nodes = 1000000;
    int num_queries = 10000;
    int num_training = 100000;
    int num_training_generated = 1000000;  // Used in generate_training.cpp

    // Interpreted File Setup
    std::string dataset_prefix = "./exports/" + dataset + "/" + dataset;
    std::string load_file = dataset_prefix + "_base.fvecs";
    std::string query_file =  dataset == "deep" || dataset == "gist" ? dataset_prefix + "_learn.fvecs" : dataset_prefix + "_query.fvecs";
    std::string groundtruth_file = num_nodes < 1000000 ? "" : dataset == "deep" || dataset == "gist" || dataset == "glove" || dataset == "sift"  ? dataset_prefix + "_groundtruth_10000.ivecs" : dataset_prefix + "_groundtruth.ivecs";
    std::string training_file = dataset_prefix + "_learn.fvecs";
    std::string generated_training_file = dataset_prefix + "_learn_1M.fvecs";
    std::string loaded_info_file = std::regex_replace(std::regex_replace(loaded_graph_file, std::regex("graph"), "info"), std::regex("bin"), "txt");
    std::string oracle_file = std::regex_replace(std::regex_replace(loaded_graph_file, std::regex("graph"), "oracle"), std::regex("bin"), "txt");

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
    // int ef_construction = 125;
    int ef_search = 400;
    int ef_search_upper = 1;
    int k_upper = 1;

    // Termination Parameters
    const bool use_distance_termination = true;
    const bool use_hybrid_termination = false; 
    const bool use_latest = false;  // Only used if use_hybrid_termination = true
    const bool use_break = false;  // Only used if use_hybrid_termination = true
    const bool use_median_break = false; // Only used if use_break = true
    const bool use_median_equations = false; //Only used if use_hybrid_termination = true
    const bool use_calculation_termination = false;
    const bool use_groundtruth_termination = false;
    const bool use_calculation_oracle = false;
    int calculations_per_query = 100000;  // Only used if use_calculation_termination = true
    int oracle_termination_total = 10000;  // Only used if use_calculation_oracle = true
    float termination_alpha = 0.4;  // Only used if use_distance_termination = true
    float alpha_break = 1.5;  // Only used if use_break = true
    float efs_break = 1.5;  // Only used if use_break = true


    std::vector<float> benchmark_median_percentiles = {0.5,0.75,0.8,0.85,0.9,0.95,0.99};
    const float breakMedian = 0.9;     //only used if use_median_break = true

    //Metric Parameters 
    const int cand_out_step = 1000; 



    //Vamana Values 
    int R = 70;
    float alpha_vamana = 2;

    // Beam-Width to Alpha Conversions using average distance
    const std::map<std::string, std::pair<float, float>> bw = !use_median_equations ? 
    std::map<std::string, std::pair<float, float>> {   //avg values 
        {"deep", {0.197, -300.85}},
        {"deep96", {0.225, -319.3}},
        {"sift", {0.2108, -339.64}},
        {"gist", {0.1114, -414.44}},
        {"glove", {0.3663, -15.865}}
    }: 
    //double check 
    std::map<std::string, std::pair<float, float>> { //median values 
        {"deep", { 0.1991,  - 319.01}}, 
        {"deep96", {0.1991, - 319.01}},
        {"sift", { 0.2068,  - 357.74}},
        {"gist", {0.1991, - 319.01}},
        {"glove", {0.3687, - 6.2159}}

    };


    const std::map<std::string, std::pair<float, float>> alpha = !use_median_equations ? 
    std::map<std::string, std::pair<float, float>>{ //avg values 
        {"50 deep", {0.0185, 0.2273}}, 
        {"10 deep", {0.0169, 0.2548}},  
        {"1 deep", {0.0151, 0.2857}},
        {"50 sift", {0.0269f, 0.1754}},
        {"10 sift", {0.0244, 0.2155}},
        {"1 sift", {0.0222, 0.2558}},
        {"50 gist", {0.013f, 0.2454}}, 
        {"10 gist", {0.0111f, 0.2707}},
        {"1 gist", {0.0093, 0.2964}},
        {"50 glove", {0.0191, 0.2111}}, 
        {"10 glove", {0.0188, 0.216}},
        {"1 glove", {0.0184, 0.222}},
        {"50 deep96", {0.0215, 0.2152}}, 
        {"10 deep96", {0.0199, 0.2438}},
        {"1 deep96", {0.0184, 0.2759}}
    }: 
    std::map<std::string, std::pair<float, float>> //median values 
    {
        {"50 deep", {0.0215, 0.2096}}, 
        {"10 deep", {0.0196, 0.2417}},   
        {"1 deep", {0.0176, 0.2811}}, 
        {"50 sift", {0.029, 0.166}},   
        {"10 sift", {0.0266, 0.2077}}, 
        {"1 sift", {0.0253,  0.2517}}, 
        {"50 gist", {0.0133, 0.2445}}, 
        {"10 gist", { 0.0115, 0.2706}},  
        {"1 gist", {0.0176, 0.2454}},  
        {"50 glove", {0.0186, 0.2238}}, 
        {"10 glove", {0.0185, 0.2263}},  
        {"1 glove", {0.0184, 0.2287}},  
        {"50 deep96", {0.0296, 0.1602}},  
        {"10 deep96", {0.0273, 0.2013}},  
        {"1 deep96", {0.0176, 0.2811}}  

    };
    std::string alpha_key = std::to_string(num_return) + " " + dataset;
    float bw_slope = use_hybrid_termination ? bw.at(dataset).first : 0; 
    float bw_intercept = use_hybrid_termination ? bw.at(dataset).second : 0;
    float alpha_coefficient = use_hybrid_termination ? alpha.at(alpha_key).first : 0;
    float alpha_intercept = use_hybrid_termination ? alpha.at(alpha_key).second : 0;

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
    int initial_cost = 0;
    int initial_benefit = 0;
    
    // Grid parameters: repeat all benchmarks for each set of grid values
    std::vector<int> grid_num_return = {};
    // {1 ,10 , 50 };
    std::vector<std::string> grid_runs_prefix = {};
    // {"./runs_fall_2024/run/"+ dataset+"/distance_termination/k=1_distance_termination_3.1", "./runs_fall_2024/run/"+ dataset+"/distance_termination/k=10_distance_termination_3.1", "./runs_fall_2024/run/"+ dataset+"/distance_termination/k=50_distance_termination_3.1" };
    std::vector<std::string> grid_graph_file = {};
    // {loaded_graph_file,loaded_graph_file,loaded_graph_file};

    
    // Benchmark parameters
    std::vector<int> benchmark_num_return = {};
    std::vector<int> benchmark_optimal_connections = {};
    std::vector<int> benchmark_max_connections = {};
    std::vector<int> benchmark_max_connections_0 = {};
    std::vector<int> benchmark_ef_construction = {};
    // std::vector<int> benchmark_ef_search  = {200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000,4500, 5000}; 
    std::vector<int> benchmark_ef_search = { };
    std::vector<float> benchmark_termination_alpha ={0.35,0.36,0.37,0.38,0.39,0.40, 0.41, 0.42, 0.43, 0.44, 0.45};
    // {0.1,0.11, 0.12, 0.13, 0.14, 0.15, 0.16,0.17, 0.18,0.19, 0.2, 0.21, 0.22, 0.23}; 
    // {0.03,0.04,0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.11};
    // {0.14, 0.15, 0.16,0.17, 0.18,0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25,0.26, 0.27, 0.28, 0.29, 0.30, 0.32, 0.34, 0.35};
    // {0.12, 0.13, 0.14, 0.15, 0.16,0.17, 0.18,0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25,0.275 };

    // {0.03,0.04,0.05, 0.06, 0.07, 0.08,0.09, 0.1, 0.11, 0.12,0.14, 0.15};
    // {0.15, 0.175, 0.2, 0.23, 0.24, 0.25 ,0.275, 0.3, 0.325, 0.35};
    // {0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.13, 0.14, 0.15, 0.16};
    // {0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1};
    std::vector<float> benchmark_learning_rate = {};
    std::vector<float> benchmark_initial_temperature = {};
    std::vector<float> benchmark_decay_factor = {};
    std::vector<float> benchmark_initial_keep_ratio = {};
    std::vector<float> benchmark_final_keep_ratio = {};
    std::vector<float> benchmark_stinky_points = {};
    std::vector<int> benchmark_grasp_loops = {};
    std::vector<int> benchmark_grasp_subloops = {};
    std::vector<int> benchmark_calculations_per_query = {};
    std::vector<int> benchmark_oracle_termination_total = {};
    std::vector<float> benchmark_alpha_vamana = {};

   
    // Debugging Flags
    const bool export_benchmark = true;
    const bool export_median_calcs = false;  // Replaces mean with median in benchmark file
    const bool export_median_precentiles = false;
    const bool export_graph = false;
    const bool export_histograms = false;
    const bool export_candidate_popping_times = false;
    const bool export_weight_updates = false;
    const bool export_oracle = false;  // Log distance calcs needed to find exact nearest neighbors
    const bool export_clustering_coefficient = false;
    const bool export_cost_benefit_pruned = false;  // Log edges pruned by cost-benefit training
    const bool export_calcs_per_query = false;  // Log distance calcs performed during search
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
    int interval_for_calcs_histogram = 1000;

    // Generation Settings
    std::string training_set = "";
    const bool generate_our_training = false;
    const bool regenerate_each_iteration = false;
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

    // Parameters For dataset_metrics.cpp
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
