#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include "hnsw.h"

using namespace std;

void knn_search(Config* config, vector<vector<int>>& actual_neighbors, float** nodes, float** queries) {
    bool use_groundtruth = config->groundtruth_file != "";
    if (use_groundtruth && config->query_file == "") {
        cout << "Warning: Groundtruth file will not be used because queries were generated" << endl;
        use_groundtruth = false;
    }
    if (use_groundtruth) {
        // Load actual nearest neighbors
        load_ivecs(config->groundtruth_file, actual_neighbors, config->num_queries, config->num_return);

        if (config->benchmark_print_neighbors) {
            for (int i = 0; i < config->num_queries; ++i) {
                cout << "Neighbors in ideal case for query " << i << endl;
                for (size_t j = 0; j < actual_neighbors[i].size(); ++j) {
                    float dist = calculate_l2_sq(queries[i], nodes[actual_neighbors[i][j]], config->dimensions, -1);
                    cout << actual_neighbors[i][j] << " (" << dist << ") ";
                }
                cout << endl;
            }
        }
    } else {
        // Calcuate actual nearest neighbors per query
        auto start = chrono::high_resolution_clock::now();
        actual_neighbors.resize(config->num_queries);
        for (int i = 0; i < config->num_queries; ++i) {
            priority_queue<pair<float, int>> pq;

            for (int j = 0; j < config->num_nodes; ++j) {
                float dist = calculate_l2_sq(queries[i], nodes[j], config->dimensions, -1);
                pq.emplace(dist, j);
                if (pq.size() > config->num_return)
                    pq.pop();
            }

            // Place actual nearest neighbors
            actual_neighbors[i].resize(config->num_return);

            size_t idx = pq.size();
            while (idx > 0) {
                --idx;
                actual_neighbors[i][idx] = pq.top().second;
                pq.pop();
            }

            // Print out neighbors
            if (config->benchmark_print_neighbors) {
                cout << "Neighbors in ideal case for query " << i << endl;
                for (size_t j = 0; j < actual_neighbors[i].size(); ++j) {
                    float dist = calculate_l2_sq(queries[i], nodes[actual_neighbors[i][j]], config->dimensions, -1);
                    cout << actual_neighbors[i][j] << " (" << dist << ") ";
                }
                cout << endl;
            }
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Brute force time: " << duration / 1000.0 << " seconds" << endl;
    }
}

void run_benchmark(Config* config, int& parameter, const vector<int>& parameter_values, const string& parameter_name,
        float** nodes, float** queries, ofstream* results_file) {

    int default_parameter = parameter;
    if (config->export_benchmark_hnsw) {
        *results_file << "\nVarying " << parameter_name;
    }

    for (int i = 0; i < parameter_values.size(); i++) {
        parameter = parameter_values[i];
        if (config->export_benchmark_hnsw) {
            *results_file << endl << parameter << ", ";
        }
        // Sanity checks
        if(!config->sanity_checks()) {
            cout << "Config error!" << endl;
            break;
        }

        layer0_dist_comps = 0;
        upper_dist_comps = 0;
        vector<vector<pair<float, int>>> neighbors;
        double construction_duration;
        double search_duration;
        long long search_dist_comp;
        HNSW* hnsw = NULL;
        vector<vector<int>> actual_neighbors;
        knn_search(config, actual_neighbors, nodes, queries);

        if (config->load_graph_file) {
            hnsw = init_hnsw(config, nodes);
            load_hnsw_file(config, hnsw, nodes, true);
        } else {
            // Insert nodes into HNSW
            auto start = chrono::high_resolution_clock::now();

            cout << "Benchmarking with parameters: "
                << "opt_con = " <<  config->optimal_connections << ", max_con = "
                << config->max_connections << ", max_con_0 = " << config->max_connections_0
                << ", ef_construction = " << config->ef_construction << ", ef_search = "
                << config->ef_search << ", num_return = " << config->num_return << endl; 
            hnsw = init_hnsw(config, nodes);
            for (int i = 1; i < config->num_nodes; i++) {
                hnsw->insert(config, i);
            }

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            cout << "Construction time: " << duration / 1000.0 << " seconds, ";
            cout << "Distance computations (layer 0): " << layer0_dist_comps << ", ";
            cout << "Distance computations (top layers): " << upper_dist_comps << endl;
            construction_duration = duration / 1000.0;
        }

        if (config->ef_search < config->num_return) {
            cout << "Warning: Skipping ef_search = " << config->ef_search << " which is less than num_return" << endl;
            search_duration = 0;
            search_dist_comp = 0;
            break;
        }

        // Run query search
        auto start = chrono::high_resolution_clock::now();
        layer0_dist_comps = 0;
        upper_dist_comps = 0;
        neighbors.reserve(config->num_queries);
        vector<vector<Edge*>> path;
        for (int i = 0; i < config->num_queries; ++i) {
            pair<int, float*> query = make_pair(i, queries[i]);
            neighbors.emplace_back(hnsw->nn_search(config, path, query, config->num_return));
        }

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Query time: " << duration / 1000.0 << " seconds, ";
        cout << "Distance computations (layer 0): " << layer0_dist_comps << ", ";
        cout << "Distance computations (top layers): " << upper_dist_comps << endl;

        search_duration = duration;
        search_dist_comp = layer0_dist_comps;

        if (neighbors.empty())
            break;

        cout << "Results for construction parameters: " << config->optimal_connections << ", " << config->max_connections << ", "
            << config->max_connections_0 << ", " << config->ef_construction << " and search parameters: " << config->ef_search << endl;

        int similar = 0;
        for (int j = 0; j < config->num_queries; ++j) {
            // Find similar neighbors
            unordered_set<int> actual_set(actual_neighbors[j].begin(), actual_neighbors[j].end());
            unordered_set<int> intersection;

            for (size_t k = 0; k < neighbors[j].size(); ++k) {
                auto n_pair = neighbors[j][k];
                if (actual_set.find(n_pair.second) != actual_set.end()) {
                    intersection.insert(n_pair.second);
                }
            }
            similar += intersection.size();

            // Print out neighbors[i][j]
            if (config->benchmark_print_neighbors) {
                cout << "Neighbors for query " << j << ": ";
                for (size_t k = 0; k < neighbors[j].size(); ++k) {
                    auto n_pair = neighbors[j][k];
                    cout << n_pair.second << " (" << n_pair.first << ") ";
                }
                cout << endl;
            }

            // Print missing neighbors between intersection and actual_neighbors
            if (config->benchmark_print_missing) {
                cout << "Missing neighbors for query " << j << ": ";
                if (intersection.size() == actual_neighbors[j].size()) {
                    cout << "None" << endl;
                    continue;
                }
                for (size_t k = 0; k < actual_neighbors[j].size(); ++k) {
                    if (intersection.find(actual_neighbors[j][k]) == intersection.end()) {
                        float dist = calculate_l2_sq(queries[j], nodes[actual_neighbors[j][k]], config->dimensions, -1);
                        cout << actual_neighbors[j][k] << " (" << dist << ") ";
                    }
                }
                cout << endl;
            }
        }

        double recall = (double) similar / (config->num_queries * config->num_return);
        cout << "Correctly found neighbors: " << similar << " ("
            << recall * 100 << "%)" << endl;

        if (config->export_benchmark_hnsw) {
            *results_file << search_dist_comp / config->num_queries << ", "
            << recall << ", " << search_duration / config->num_queries << ", " << construction_duration;
        }

        delete hnsw;
    }
    if (config->export_benchmark_hnsw) {
        *results_file << endl;
    }
    parameter = default_parameter;
}

/**
 * This class is used to run HNSW with different parameters, comparing the recall
 * versus ideal for each set of parameters.
*/
int main() {
    time_t now = time(0);
    cout << "Benchmark run started at " << ctime(&now);
    Config* config = new Config();

    // Get graph nodes and queries
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** queries = new float*[config->num_queries];
    load_queries(config, nodes, queries);

    // Initialize output file
    ofstream* results_file = NULL;
    if (config->export_benchmark_hnsw) {
        results_file = new ofstream(config->benchmark_file);
        *results_file << "Size " << config->num_nodes  << ", CPU TYPE  "  << CPUBrandString << "\nDefault Parameters: opt_con = "
            << config->optimal_connections << ", max_con = " << config->max_connections << ", max_con_0 = " << config->max_connections_0
            << ", ef_con = " << config->ef_construction << ", scaling_factor = " << config->scaling_factor
            << ", ef_search = " << config->ef_search << ", num_return = " << config->num_return
            << "\nparameter, dist_comps/query, recall, runtime/query (ms)" << endl;
    }

    // Run benchmarks
    run_benchmark(config, config->optimal_connections, config->benchmark_optimal_connections,
        "Optimal Connections:", nodes, queries, results_file);
    run_benchmark(config, config->max_connections, config->benchmark_max_connections,
        "Max Connections:", nodes, queries, results_file);
    run_benchmark(config, config->max_connections_0, config->benchmark_max_connections_0,
        "Max Connections 0:", nodes, queries, results_file);
    run_benchmark(config, config->ef_construction, config->benchmark_ef_construction,
        "ef Construction:", nodes, queries, results_file);
    run_benchmark(config, config->ef_search, config->benchmark_ef_search, "ef Search:", nodes,
        queries, results_file);
    run_benchmark(config, config->num_return, config->benchmark_num_return, "Num Return:",
        nodes, queries, results_file);

    // Clean up
    if (results_file != NULL) {
        results_file->close();
        delete results_file;
        cout << "Results exported to " << config->benchmark_file << endl;
    }
    for (int i = 0; i < config->num_nodes; i++)
        delete[] nodes[i];
    delete[] nodes;
    for (int i = 0; i < config->num_queries; ++i)
        delete[] queries[i];
    delete[] queries;
    delete config;

    // Print time elapsed
    now = time(0);
    cout << "Benchmark run ended at " << ctime(&now);
}
