#include <iostream>
#include <chrono>
#include "grasp.h"
#include "../HNSW/hnsw.h"

using namespace std;

int main() {
    // Initialize time and config
    auto begin_time = chrono::high_resolution_clock::now();
    time_t now = time(NULL);
    cout << "GraSP run started at " << ctime(&now);
    Config* config = new Config();


     cout   << "Size " << config->num_nodes << "\nDefault Parameters: opt_con = "
            << config->optimal_connections << ", max_con = " << config->max_connections << ", max_con_0 = " << config->max_connections_0
            << ", ef_con = " << config->ef_construction << ", scaling_factor = " << config->scaling_factor
            << ", ef_search = " << config->ef_search << ", num_return = " << config->num_return
            << ", learning_rate = " << config->learning_rate << ", initial_temperature = " << config->initial_temperature
            << ", decay_factor = " << config->decay_factor << ", initial_keep_ratio = " << config->initial_keep_ratio
            << ", final_keep_ratio = " << config->final_keep_ratio << ", grasp_loops = " << config->grasp_loops  

            <<"\nCurrent Run Properties: Stinky Values = "  << std::boolalpha  <<  config->use_stinky_points << " [" <<config->stinkyValue <<"]" 
            << ", use_heuristic = " << config->use_heuristic << ", use_dynamic_sampling = " << config->use_dynamic_sampling 
            << ", Single search point = " << config->single_entry_point  << ", current Pruning method = " << config->weight_selection_method   << endl;

    // Clear histogram file if it exists
    if (!config->histogram_prob_file.empty()) {
        ofstream histogram = ofstream(config->histogram_prob_file);
        histogram.close();
    }

    // Load nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** queries = new float*[config->num_queries];
    load_queries(config, nodes, queries);
    float** training = new float*[config->num_training];
    load_training(config, nodes, training);
    remove_duplicates(config, training, queries);
    
    // Construct HNSW
    cout << "Beginning HNSW construction" << endl;
    HNSW* hnsw = init_hnsw(config, nodes);
    if (config->load_graph_file) {
        load_hnsw_file(config, hnsw, nodes);
    } else {
        for (int i = 1; i < config->num_nodes; i++) {
            hnsw->insert(config, i);
        }
    }

    // Optimize HNSW using GraSP
    vector<Edge*> edges = hnsw->get_layer_edges(config, 0);
    cout << "Starting Edges: " << edges.size() << endl;
    learn_edge_importance(config, hnsw, edges, training);
    prune_edges(config, hnsw, edges, config->final_keep_ratio * edges.size());
    edges = hnsw->get_layer_edges(config, 0);
    cout << "Final Edges: " << edges.size() << endl;
    for (int i = 0; i < config->num_training; i++)
        delete[] training[i];
    delete[] training;

    // Run queries
    if (config->run_search) {
        auto search_start = chrono::high_resolution_clock::now();
        cout << "Time passed: " << chrono::duration_cast<chrono::milliseconds>(search_start - begin_time).count() << " ms" << endl;
        cout << "Beginning search" << endl;

        // Run query search and print results
        hnsw->search_queries(config, queries);

        auto search_end = chrono::high_resolution_clock::now();
        cout << "Time passed: " << chrono::duration_cast<chrono::milliseconds>(search_end - search_start).count() << " ms" << endl;

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
    cout << "GraSP run ended at " << ctime(&now);
    auto end_time = chrono::high_resolution_clock::now();
    cout << "Total time taken: " << chrono::duration_cast<chrono::milliseconds>(end_time - begin_time).count() << " ms" << endl;

    return 0;
}
