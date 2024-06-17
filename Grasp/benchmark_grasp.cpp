// #include <iostream>
// #include <chrono>
// #include "grasp.h"
// #include "../HNSW/hnsw.h"

// using namespace std;

// //Parameters: learninb rate, starting temprature, decay factor, keep ratio, max iteration, cadidate queue size 




// void knn_search(Config* config, vector<vector<int>>& actual_neighbors, float** nodes, float** queries) {
//     bool use_groundtruth = config->groundtruth_file != "";
//     if (use_groundtruth && config->query_file == "") {
//         cout << "Warning: Groundtruth file will not be used because queries were generated" << endl;
//         use_groundtruth = false;
//     }
//     if (use_groundtruth) {
//         // Load actual nearest neighbors
//         load_ivecs(config->groundtruth_file, actual_neighbors, config->num_queries, config->num_return);

//         if (config->benchmark_print_neighbors) {
//             for (int i = 0; i < config->num_queries; ++i) {
//                 cout << "Neighbors in ideal case for query " << i << endl;
//                 for (size_t j = 0; j < actual_neighbors[i].size(); ++j) {
//                     float dist = calculate_l2_sq(queries[i], nodes[actual_neighbors[i][j]], config->dimensions, -1);
//                     cout << actual_neighbors[i][j] << " (" << dist << ") ";
//                 }
//                 cout << endl;
//             }
//         }
//     } else {
//         // Calcuate actual nearest neighbors per query
//         auto start = chrono::high_resolution_clock::now();
//         actual_neighbors.resize(config->num_queries);
//         for (int i = 0; i < config->num_queries; ++i) {
//             priority_queue<pair<float, int>> pq;

//             for (int j = 0; j < config->num_nodes; ++j) {
//                 float dist = calculate_l2_sq(queries[i], nodes[j], config->dimensions, -1);
//                 pq.emplace(dist, j);
//                 if (pq.size() > config->num_return)
//                     pq.pop();
//             }

//             // Place actual nearest neighbors
//             actual_neighbors[i].resize(config->num_return);

//             size_t idx = pq.size();
//             while (idx > 0) {
//                 --idx;
//                 actual_neighbors[i][idx] = pq.top().second;
//                 pq.pop();
//             }

//             // Print out neighbors
//             if (config->benchmark_print_neighbors) {
//                 cout << "Neighbors in ideal case for query " << i << endl;
//                 for (size_t j = 0; j < actual_neighbors[i].size(); ++j) {
//                     float dist = calculate_l2_sq(queries[i], nodes[actual_neighbors[i][j]], config->dimensions, -1);
//                     cout << actual_neighbors[i][j] << " (" << dist << ") ";
//                 }
//                 cout << endl;
//             }
//         }
//         auto end = chrono::high_resolution_clock::now();
//         auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
//         cout << "Brute force time: " << duration / 1000.0 << " seconds" << endl;
//     }
// }





// void run_benchmark(Config* config, int& parameter, const vector<int>& parameter_values, const string& parameter_name,
//         float** nodes, float** queries, ofstream* results_file) {

//     int default_parameter = parameter;
//     if (config->export_benchmark) {
//         *results_file << "\nVarying " << parameter_name;
//     }

//     for (int i = 0; i < parameter_values.size(); i++) {
//         parameter = parameter_values[i];
//         if (config->export_benchmark) {
//             *results_file << endl << parameter << ", ";
//         }
//         // Sanity checks
//         if(!config->sanity_checks()) {
//             cout << "Config error!" << endl;
//             break;
//         }

//         layer0_dist_comps = 0;
//         vector<vector<pair<float, int>>> neighbors;
//         double search_duration;
//         long long search_dist_comp;
//         HNSW* hnsw = NULL;
//         vector<vector<int>> actual_neighbors;
//         knn_search(config, actual_neighbors, nodes, queries);

//         if (config->load_graph_file) {
//             hnsw = init_hnsw(config, nodes);
//             load_hnsw_file(config, hnsw, nodes, true);
//         } else {
//             // Insert nodes into HNSW
//             auto start = chrono::high_resolution_clock::now();

//             cout << "Benchmarking with parameters: "
//                 << "opt_con = " <<  config->optimal_connections << ", max_con = "
//                 << config->max_connections << ", max_con_0 = " << config->max_connections_0
//                 << ", ef_construction = " << config->ef_construction << ", ef_search = "
//                 << config->ef_search << ", num_return = " << config->num_return << endl; 
//             hnsw = init_hnsw(config, nodes);
//             for (int i = 1; i < config->num_nodes; i++) {
//                 hnsw->insert(config, i);
//             }

//             auto end = chrono::high_resolution_clock::now();

// void run_benchmark(){
    
    
//     cout << "Beginning HNSW construction" << endl;
//     HNSW* hnsw = init_hnsw(config, nodes);
//     if (config->load_graph_file) {
//         load_hnsw_file(config, hnsw, nodes);
//     } else {
//         for (int i = 1; i < config->num_nodes; i++) {
//             hnsw->insert(config, i);
//         }
//     }

//     // Optimize HNSW using GraSP
//     vector<Edge*> edges = hnsw->get_layer_edges(config, 0);
//     cout << "Edges: " << edges.size() << endl;
//     learn_edge_importance(config, hnsw, edges, nodes, queries);
//     prune_edges(config, hnsw, edges, config->final_keep_ratio * edges.size());
//     edges = hnsw->get_layer_edges(config, 0);
//     cout << "Edges: " << edges.size() << endl;

//     // Run queries
//     if (config->run_search) {
//         // Generate num_queries amount of queries
//         float** queries = new float*[config->num_queries];
//         load_queries(config, nodes, queries);
//         auto search_start = chrono::high_resolution_clock::now();
//         cout << "Time passed: " << chrono::duration_cast<chrono::milliseconds>(search_start - begin_time).count() << " ms" << endl;
//         cout << "Beginning search" << endl;

//         // Run query search and print results
//         hnsw->search_queries(config, queries);

//         auto search_end = chrono::high_resolution_clock::now();
//         cout << "Time passed: " << chrono::duration_cast<chrono::milliseconds>(search_end - search_start).count() << " ms" << endl;

//         // Delete queries
//         for (int i = 0; i < config->num_queries; ++i)
//             delete queries[i];
//         delete[] queries;
//     }

//     // Clean up
//     for (int i = 0; i < config->num_nodes; i++)
//         delete nodes[i];
//     delete[] nodes;
//     delete hnsw;
//     delete config;

//     // Print time elapsed
//     now = time(NULL);
//     cout << "GraSP run ended at " << ctime(&now);
//     auto end_time = chrono::high_resolution_clock::now();
//     cout << "Total time taken: " << chrono::duration_cast<chrono::milliseconds>(end_time - begin_time).count() << " ms" << endl;

// }


// int main() {
//      // Initialize time and config
//     auto begin_time = chrono::high_resolution_clock::now();
//     time_t now = time(NULL);
//     cout << "Benchmark run started at" << ctime(&now);
//     Config* config = new Config();

//     float** nodes = new float*[config->num_nodes];
//     load_nodes(config, nodes);
//     float** queries = new float*[config->num_queries];
//     load_queries(config, nodes, queries);
    
//    run_benchmark(config, config->num_return, config->benchmark_num_return, ":",
//         nodes, queries);


// }
