#include <iostream>
#include <chrono>
#include <cpuid.h>
#include <string.h>
#include "grasp.h"
#include "hnsw.h"

using namespace std;


int main() {
    // Initialize time and config
    auto begin_time = chrono::high_resolution_clock::now();
    time_t now = time(NULL);
    cout << "Run started at " << ctime(&now);
    Config* config = new Config();

    cout << "Size " << config->num_nodes << "\nBenchmarking with Parameters: opt_con = "
        << config->optimal_connections << ", max_con = " << config->max_connections << ", max_con_0 = " << config->max_connections_0
        << ", ef_con = " << config->ef_construction << ", scaling_factor = " << config->scaling_factor
        << ", ef_search = " << config->ef_search << "\nnum_return = " << config->num_return
        << ", learning_rate = " << config->learning_rate << ", initial_temperature = " << config->initial_temperature
        << ", decay_factor = " << config->decay_factor << ", initial_keep_ratio = " << config->initial_keep_ratio
        << ", final_keep_ratio = " << config->final_keep_ratio << ", grasp_loops = " << config->grasp_loops  
        <<"\nCurrent Run Properties: Stinky Values = "  << std::boolalpha  <<  config->use_stinky_points << " [" <<config->stinky_value <<"]" 
        << ", use_heuristic = " << config->use_heuristic << ", use_grasp = " << config->use_grasp << ", use_dynamic_sampling = " << config->use_dynamic_sampling 
        << ", Single search point = " << config->single_ep_construction  << ", current Pruning method = " << config->weight_selection_method   
        << "\nUse_distance_termination = " << config->use_distance_termination << ", use_cost_benefit = " << config->use_cost_benefit 
        << ", use_direct_path = " << config->use_direct_path << endl;
        
    // Clear histogram files if they exist
    if (config->export_histograms && !config->load_graph_file) {
        if (config->use_grasp) {
            ofstream histogram = ofstream(config->runs_prefix + "histogram_prob.txt");
            histogram.close();
            histogram = ofstream(config->runs_prefix + "histogram_weights.txt");
            histogram.close();
            histogram = ofstream(config->runs_prefix + "histogram_edge_updates.txt");
            histogram.close();
        }
        if (config->use_cost_benefit) {
            ofstream histogram = ofstream(config->runs_prefix + "histogram_cost.txt");
            histogram.close();
            histogram = ofstream(config->runs_prefix + "histogram_benefit.txt");
            histogram.close();
        }
    }

    // Load nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** queries = new float*[config->num_queries];
    load_queries(config, nodes, queries);
    
    // Construct HNSW
    cout << "Beginning HNSW construction" << endl;
    HNSW* hnsw = new HNSW(config, nodes);
    if (config->load_graph_file) {
        hnsw->from_files(config, nodes);
    } else {
        for (int i = 1; i < config->num_nodes; i++) {
            hnsw->insert(config, i);
        }

    }

    // Run queries
    if (config->run_search) {
   
 
        // Run query search and print results
        hnsw->search_queries_logging_datatypes(config, queries, 0.05);

  
        if (config->print_path_size) {
            cout << "Average Path Size: " << static_cast<double>(hnsw->total_path_size) / config->num_queries << endl;
            hnsw->total_path_size = 0;
        }

        // Delete queries
        for (int i = 0; i < config->num_queries; ++i)
            delete queries[i];
        delete[] queries;
    }

    // Clean up
    for (int i = 0; i < config->num_nodes; i++)
        delete[] nodes[i];
    delete[] nodes;
    delete hnsw;
    delete config;

    // Print time elapsed
    now = time(NULL);
    cout << "run ended at " << ctime(&now);
    auto end_time = chrono::high_resolution_clock::now();
    cout << "Total time taken: " << chrono::duration_cast<chrono::milliseconds>(end_time - begin_time).count() << " ms" << endl;

    return 0;
}
