#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <cpuid.h>
#include <string.h>
#include "grasp.h"
#include "hnsw.h"

using namespace std;

void get_actual_neighbors(Config* config, vector<vector<int>>& actual_neighbors, float** nodes, float** queries) {
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
                    float dist = calculate_l2_sq(queries[i], nodes[actual_neighbors[i][j]], config->dimensions);
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
                float dist = calculate_l2_sq(queries[i], nodes[j], config->dimensions);
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
                    float dist = calculate_l2_sq(queries[i], nodes[actual_neighbors[i][j]], config->dimensions);
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

template <typename T>
void run_benchmark(Config* config, T& parameter, const vector<T>& parameter_values, const string& parameter_name,
        float** nodes, float** queries, float** training, ofstream* results_file) {

    if (parameter_values.empty()) {
        return;
    }
    T default_parameter = parameter;
    if (config->export_benchmark) {
        *results_file << "\nVarying " << parameter_name << ":" << endl;
    }
    vector<std::string> lines;
    for (int i = 0; i < parameter_values.size(); i++) {
        parameter = parameter_values[i];
        // Sanity checks
        if(!config->sanity_checks()) {
            cout << "Config error!" << endl;
            break;
        }

        vector<vector<pair<float, int>>> neighbors;
        double construction_duration;
        double search_duration;
        long long search_dist_comp;
        long long total_dist_comp;
        HNSW* hnsw = NULL;
        vector<vector<int>> actual_neighbors;
        get_actual_neighbors(config, actual_neighbors, nodes, queries);

        if (config->load_graph_file) {
            hnsw = init_hnsw(config, nodes);
            load_hnsw_files(config, hnsw, nodes, true);
        } else {
            // Insert nodes into HNSW
            auto start = chrono::high_resolution_clock::now();

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
                 << "\nUse_distance_termination = " << config->use_distance_termination << ", use_benefit_cost = " << config->use_benefit_cost 
                 << ", use_direct_path = " << config->use_direct_path << endl;


                


            hnsw = init_hnsw(config, nodes);
            for (int i = 1; i < config->num_nodes; ++i) {
                hnsw->insert(config, i);
            }

            // Run GraSP
            if (config->use_grasp) {
                vector<Edge*> edges = hnsw->get_layer_edges(config, 0);
                learn_edge_importance(config, hnsw, edges, training, results_file);
                prune_edges(config, hnsw, edges, config->final_keep_ratio * edges.size());
                edges = hnsw->get_layer_edges(config, 0);
                if (config->export_histograms) {
                    ofstream histogram = ofstream(config->runs_prefix + "histogram_prob.txt", std::ios::app);
                    histogram << endl;
                    histogram.close();
                    histogram = ofstream(config->runs_prefix + "histogram_weights.txt", std::ios::app);
                    histogram << endl;
                    histogram.close();
                    histogram = ofstream(config->runs_prefix + "histogram_edge_updates.txt", std::ios::app);
                    histogram << endl;
                    histogram.close();
                }
            }

            if (config->use_benefit_cost) {
                vector<Edge*> edges = hnsw->get_layer_edges(config, 0);
                learn_cost_benefit(config, hnsw, edges, training, config->final_keep_ratio * edges.size());
                if (config->export_histograms) {
                    ofstream histogram = ofstream(config->runs_prefix + "histogram_cost.txt", std::ios::app);
                    histogram << endl;
                    histogram.close();
                    histogram = ofstream(config->runs_prefix + "histogram_benefit.txt", std::ios::app);
                    histogram << endl;
                    histogram.close();
                }
            }

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            cout << "Construction time: " << duration / 1000.0 << " seconds, ";
            cout << "Distance computations (layer 0): " << hnsw->layer0_dist_comps << ", ";
            cout << "Distance computations (top layers): " << hnsw->upper_dist_comps << endl;
            construction_duration = duration / 1000.0;
        }

        if (config->ef_search < config->num_return) {
            cout << "Warning: Skipping ef_search = " << config->ef_search << " which is less than num_return" << endl;
            search_duration = 0;
            search_dist_comp = 0;
            total_dist_comp = 0;
            break;
        }

        // Run query search
        hnsw->reset_statistics();
        auto start = chrono::high_resolution_clock::now();
        neighbors.reserve(config->num_queries);
        vector<Edge*> path;
        for (int i = 0; i < config->num_queries; ++i) {
            pair<int, float*> query = make_pair(i, queries[i]);
            neighbors.emplace_back(hnsw->nn_search(config, path, query, config->num_return));
            if (config->print_neighbor_percent) {
                for (int i = 0; i < hnsw->percent_neighbors.size(); ++i) {
                    cout << hnsw->percent_neighbors[i] << " ";
                }
                cout << endl;
                hnsw->percent_neighbors.clear();
            }
        }

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Query time: " << duration / 1000.0 << " seconds, ";
        cout << "Distance computations (layer 0): " << hnsw->layer0_dist_comps << ", ";
        cout << "Distance computations (top layers): " << hnsw->upper_dist_comps << endl;

        search_duration = duration;
        search_dist_comp = hnsw->layer0_dist_comps;
        total_dist_comp = hnsw->layer0_dist_comps + hnsw->upper_dist_comps;

        if (neighbors.empty())
            break;

        cout << "Results for construction parameters: " << config->optimal_connections << ", " << config->max_connections << ", "
            << config->max_connections_0 << ", " << config->ef_construction << " and search parameters: " << config->ef_search << endl;

        int similar = 0;
        float total_ndcg = 0;
        for (int j = 0; j < config->num_queries; ++j) {
            // Find similar neighbors
            unordered_set<int> actual_set(actual_neighbors[j].begin(), actual_neighbors[j].end());
            unordered_set<int> intersection;
            float actual_gain = 0;
            float ideal_gain = 0;

            for (size_t k = 0; k < neighbors[j].size(); ++k) {
                auto n_pair = neighbors[j][k];
                float gain = 1 / log2(k + 2);
                ideal_gain += gain;
                if (actual_set.find(n_pair.second) != actual_set.end()) {
                    intersection.insert(n_pair.second);
                    actual_gain += gain;
                }
            }
            similar += intersection.size();
            total_ndcg += actual_gain / ideal_gain;

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
                        float dist = calculate_l2_sq(queries[j], nodes[actual_neighbors[j][k]], config->dimensions);
                        cout << actual_neighbors[j][k] << " (" << dist << ") ";
                    }
                }
                cout << endl;
            }
        }

        double recall = (double) similar / (config->num_queries * config->num_return);
        cout << "Correctly found neighbors: " << similar << " ("
            << recall * 100 << "%)" << endl;

        double average_ndcg = (double) total_ndcg / config->num_queries;
        cout << "Average NDCG@" << config->num_return << ": " << average_ndcg << endl;

        if (config->export_benchmark ) {
            std::string line = std::to_string(parameter) + ", " 
                     + std::to_string(search_dist_comp / config->num_queries) + ", "
                     + std::to_string(recall) + ", " 
                     + std::to_string(search_duration / config->num_queries) + ", " 
                     + std::to_string(construction_duration) + ", "
                     + std::to_string(total_dist_comp / config->num_queries) + ", "
                     + std::to_string(static_cast<double>(hnsw->actual_beam_width) / config->num_queries);
            lines.push_back(line);
        }
        
        if (config->export_graph && !config->load_graph_file) {
            save_hnsw_files(config, hnsw, parameter_name + "_" + to_string(parameter_values[i]), construction_duration);
        }

        delete hnsw;
    }
    if (config->export_benchmark) {
        *results_file << "\nparameter, dist_comps/query, recall, runtime/query (ms), construction time, total_dist_comps/query, actual_beam_width" << endl;
        for(auto& line: lines)
            *results_file << line <<endl;
        *results_file << endl << endl;
    }
    parameter = default_parameter;
}

string get_cpu_brand() {
    char CPUBrand[0x40];
    unsigned int CPUInfo[4] = {0,0,0,0};

    __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    unsigned int nExIds = CPUInfo[0];

    memset(CPUBrand, 0, sizeof(CPUBrand));

    for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
        __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
        if (i == 0x80000002)
            memcpy(CPUBrand, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            memcpy(CPUBrand + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            memcpy(CPUBrand + 32, CPUInfo, sizeof(CPUInfo));
    }
    string output(CPUBrand);
    return output;
}

/**
 * This class is used to run HNSW with different parameters, comparing the recall
 * versus ideal for each set of parameters.
*/
int main() {

    string CPUBrand = get_cpu_brand();

    time_t now = time(0);
    cout << "Benchmark run started at " << ctime(&now);
    Config* config = new Config();

    // Load nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** queries = new float*[config->num_queries];
    load_queries(config, nodes, queries);
    float** training = nullptr;
    if ((config->use_grasp || config->use_benefit_cost) && !config->load_graph_file) {
        training = new float*[config->num_training];
        load_training(config, nodes, training);
        remove_duplicates(config, training, queries);
    }

    // Initialize output files
    ofstream* results_file = NULL;
    if (config->export_benchmark) {
        results_file = new ofstream(config->runs_prefix + "benchmark.txt");
        *results_file << "Size " << config->num_nodes << "\nBenchmarking with Parameters: opt_con = "
                 << config->optimal_connections << ", max_con = " << config->max_connections << ", max_con_0 = " << config->max_connections_0
                 << ", ef_con = " << config->ef_construction << ", scaling_factor = " << config->scaling_factor
                 << ", ef_search = " << config->ef_search << "\nnum_return = " << config->num_return
                 << ", learning_rate = " << config->learning_rate << ", initial_temperature = " << config->initial_temperature
                 << ", decay_factor = " << config->decay_factor << ", initial_keep_ratio = " << config->initial_keep_ratio
                 << ", final_keep_ratio = " << config->final_keep_ratio << ", grasp_loops = " << config->grasp_loops  
                 <<"\nCurrent Run Properties: Stinky Values = "  << std::boolalpha  <<  config->use_stinky_points << " [" <<config->stinky_value <<"]" 
                 << ", use_heuristic = " << config->use_heuristic << ", use_grasp = " << config->use_grasp << ", use_dynamic_sampling = " << config->use_dynamic_sampling 
                 << ", Single search point = " << config->single_ep_construction  << ", current Pruning method = " << config->weight_selection_method   
                 << "\nUse_distance_termination = " << config->use_distance_termination << ", use_benefit_cost = " << config->use_benefit_cost 
                 << ", use_direct_path = " << config->use_direct_path << endl;

          

        if (config->export_histograms) {
            if (config->use_grasp) {
                ofstream histogram = ofstream(config->runs_prefix + "histogram_prob.txt");
                histogram.close();
                histogram = ofstream(config->runs_prefix + "histogram_weights.txt");
                histogram.close();
                histogram = ofstream(config->runs_prefix + "histogram_edge_updates.txt");
                histogram.close();
            }
            if (config->use_benefit_cost) {
                ofstream histogram = ofstream(config->runs_prefix + "histogram_cost.txt");
                histogram.close();
                histogram = ofstream(config->runs_prefix + "histogram_benefit.txt");
                histogram.close();
            }
        }
    }


    // Run benchmarks
    run_benchmark(config, config->optimal_connections, config->benchmark_optimal_connections,
        "opt_con", nodes, queries, training, results_file);
    run_benchmark(config, config->max_connections, config->benchmark_max_connections,
        "max_con", nodes, queries, training, results_file);
    run_benchmark(config, config->max_connections_0, config->benchmark_max_connections_0,
        "max_con_0", nodes, queries, training, results_file);
    run_benchmark(config, config->ef_construction, config->benchmark_ef_construction,
        "ef_construction", nodes, queries, training, results_file);
    run_benchmark(config, config->num_return, config->benchmark_num_return, "num_return",
        nodes, queries, training, results_file);
    
    if (config->use_distance_termination) {
        run_benchmark(config, config->termination_alpha, config->benchmark_termination_alpha, "termination_alpha", nodes,
            queries, training, results_file);
    } else {
        run_benchmark(config, config->ef_search, config->benchmark_ef_search, "ef_search", nodes,
            queries, training, results_file);
    }

    if (config->use_grasp) {
        run_benchmark(config, config->learning_rate, config->benchmark_learning_rate, "learning_rate",
            nodes, queries, training, results_file);
        run_benchmark(config, config->initial_temperature, config->benchmark_initial_temperature, "initial_temperature",
            nodes, queries, training, results_file);
        run_benchmark(config, config->decay_factor, config->benchmark_decay_factor, "decay_factor",
            nodes, queries, training, results_file);
        run_benchmark(config, config->initial_keep_ratio, config->benchmark_initial_keep_ratio, "initial_keep_ratio",
            nodes, queries, training, results_file);
        run_benchmark(config, config->final_keep_ratio, config->benchmark_final_keep_ratio, "final_keep_ratio",
            nodes, queries, training, results_file);
        run_benchmark(config, config->grasp_loops, config->benchmark_grasp_loops, "grasp_loops",
            nodes, queries, training, results_file);
        run_benchmark(config, config->use_stinky_points, config->benchmark_enablign_stinky, "stinky_points",
            nodes, queries, training, results_file);
    }

    // Clean up
    if (results_file != NULL) {
        results_file->close();
        delete results_file;
        cout << "Results exported to " << config->runs_prefix << "benchmark.txt" << endl;
    }
    for (int i = 0; i < config->num_nodes; ++i)
        delete[] nodes[i];
    delete[] nodes;
    for (int i = 0; i < config->num_queries; ++i)
        delete[] queries[i];
    delete[] queries;
    if (config->use_grasp && !config->load_graph_file)
        for (int i = 0; i < config->num_training; ++i)
            delete[] training[i];
    delete[] training;
    delete config;

    // Print time elapsed
    now = time(0);
    cout << "Benchmark run ended at " << ctime(&now);
}
