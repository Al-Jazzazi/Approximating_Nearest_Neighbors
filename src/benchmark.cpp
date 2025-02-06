#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <cpuid.h>
#include <string.h>
#include "./grasp/grasp.h"
#include "hnsw.h"

using namespace std;


template <typename T>
void run_benchmark(Config* config, T& parameter, const vector<T>& parameter_values, const string& parameter_name,
        float** nodes, float** queries, float** training, ofstream* results_file) {

    // Stop if parameter vector is empty
    if (parameter_values.empty()) {
        return;
    }

    T default_parameter = parameter;
    if (config->export_benchmark) {
        *results_file << "\nVarying " << parameter_name << ":" << endl;
    }
    vector<std::string> lines;
    // Set config to each parameter value
    for (int i = 0; i < parameter_values.size(); i++) {
        parameter = parameter_values[i];
        if(!config->sanity_checks()) {
            cout << "Config error!" << endl;
            break;
        }

        vector<vector<pair<float, int>>> neighbors;
        double construction_duration;
        double search_duration;
        long long search_dist_comp;
        long long total_dist_comp;
        long long candidates_popped;
        long long candidates_size;
        long long candidates_without_if;
        HNSW* hnsw = NULL;
        vector<vector<int>> actual_neighbors;
        get_actual_neighbors(config, actual_neighbors, nodes, queries);

        // Construct HNSW graph
        if (config->load_graph_file) {
         
            hnsw = new HNSW(config, nodes);
            hnsw->from_files(config, true);
        } else {
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
                 << "\nUse_distance_termination = " << config->use_distance_termination << ", use_cost_benefit = " << config->use_cost_benefit 
                 << ", use_direct_path = " << config->use_direct_path << endl;
            hnsw = new HNSW(config, nodes);
            for (int j = 1; j < config->num_nodes; ++j) {
                hnsw->insert(config, j);
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

            // Run cost-benefit pruning
            if (config->use_cost_benefit) {
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

            // Log construction statistics
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            cout << "Construction time: " << duration / 1000.0 << " seconds, ";
            cout << "Distance computations (layer 0): " << hnsw->layer0_dist_comps << ", ";
            cout << "Distance computations (top layers): " << hnsw->upper_dist_comps << endl;
            construction_duration = duration / 1000.0;
        }

        // Re-initialize statistics for query search
        if (config->ef_search < config->num_return) {
            cout << "Warning: Skipping ef_search = " << config->ef_search << " which is less than num_return" << endl;
            search_duration = 0;
            search_dist_comp = 0;
            total_dist_comp = 0;
            candidates_popped = 0;
            candidates_size = 0 ;
            candidates_without_if = 0;
            break;
        }
        hnsw->reset_statistics();
        if (config->print_path_size) {
            hnsw->total_path_size = 0;
        }
        int median_comps_layer0 = 0;
        int similar = 0;
        float total_ndcg = 0;
        vector<pair<int, int>> nn_calculations;

        // Conditionally use oracle to fake search
        if (config->use_calculation_oracle) {
            load_oracle(config, nn_calculations);
            int distance_left = config->oracle_termination_total;
            for (pair<int, int>& cur_found : nn_calculations ) {
                distance_left -= cur_found.first; 
                if(distance_left <= 0)
                    break; 
                similar++;
            }
            search_duration = 0;
            search_dist_comp = 0;
            total_dist_comp = 0;
            candidates_popped = 0;
            candidates_size = 0 ;
            candidates_without_if = 0;
        // Run query search
        } else {
            vector<int> counts_calcs;
            for (int j = 0; j < 20; j++) {
                counts_calcs.push_back(0);
            }
            auto start = chrono::high_resolution_clock::now();
            neighbors.reserve(config->num_queries);
            vector<Edge*> path;
            vector<long long int> dist_comps_per_q_vec;
            if(config->use_hybrid_termination || config->use_distance_termination ) 
                hnsw->calculate_termination(config);
            for (int j = 0; j < config->num_queries; ++j) {
                hnsw->cur_groundtruth = actual_neighbors[j];
                hnsw->layer0_dist_comps_per_q = 0;
                pair<int, float*> query_pair = make_pair(j, queries[j]);
                neighbors.emplace_back(hnsw->nn_search(config, path, query_pair, config->num_return));
                if (config->export_calcs_per_query) {
                    ++counts_calcs[std::min(19, hnsw->layer0_dist_comps_per_q / config->interval_for_calcs_histogram)];
                }
                if (config->export_median_calcs || config->export_median_precentiles) {
                    dist_comps_per_q_vec.push_back(hnsw->layer0_dist_comps_per_q);
                }
                if (config->print_neighbor_percent) {
                    for (int k = 0; k < hnsw->percent_neighbors.size(); ++k) {
                        cout << hnsw->percent_neighbors[k] << " ";
                    }
                    cout << endl;
                    hnsw->percent_neighbors.clear();
                }
            }

            // Log search statistics
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            cout << "Query time: " << duration / 1000.0 << " seconds, ";
            cout << "Distance computations (layer 0): " << hnsw->layer0_dist_comps << ", ";
            cout << "Distance computations (top layers): " << hnsw->upper_dist_comps << endl;
            if (config->print_path_size) {
                cout << "Average Path Size: " << static_cast<double>(hnsw->total_path_size) / config->num_queries << endl;
                hnsw->total_path_size = 0;
            }
            if (config->export_median_calcs || config->export_median_precentiles_alpha ) {
                std::sort(dist_comps_per_q_vec.begin(), dist_comps_per_q_vec.end());
                median_comps_layer0 = dist_comps_per_q_vec[dist_comps_per_q_vec.size() / 2];
            }
            if(config->export_median_precentiles){
                std::sort(dist_comps_per_q_vec.begin(), dist_comps_per_q_vec.end());
                ofstream histogram = ofstream(config->metric_prefix + "_median_percentiles.txt", std::ios::app);
                histogram << config->ef_search << ", ";
                 for (int j = 0; j < config->benchmark_median_percentiles.size(); ++j) {
                    
                    int k = static_cast<int>(dist_comps_per_q_vec.size()* config->benchmark_median_percentiles[j]);
                    // cout << "k is " << k << endl;
                    if(k < dist_comps_per_q_vec.size())
                        histogram << dist_comps_per_q_vec[k] << ", ";
                }
                histogram << endl;

            }
            if(config->export_median_precentiles_alpha && config->use_distance_termination){
                std::sort(dist_comps_per_q_vec.begin(), dist_comps_per_q_vec.end());
                ofstream histogram = ofstream("./alpha_median_percentiles/" +config->graph + "/_" +  config->dataset + "_median_percentiles_k=" +  std::to_string(config->num_return)+ "_distance_term_" + std::to_string(config->alpha_termination_selection)  + ".txt", std::ios::app);
                histogram << config->termination_alpha << ", ";
                 for (int j = 0; j < config->benchmark_median_percentiles.size(); ++j) {
                    
                    int k = static_cast<int>(dist_comps_per_q_vec.size()* config->benchmark_median_percentiles[j]);
                    cout << "k is " << k <<  " " << dist_comps_per_q_vec.size() << endl;
                    if(k < dist_comps_per_q_vec.size())
                        histogram << dist_comps_per_q_vec[k] << ", ";
                }
                histogram << endl;

            }
            if (config->export_calcs_per_query) {
                ofstream histogram = ofstream(config->runs_prefix + "histogram_calcs_per_query.txt", std::ios::app);
                for (int j = 0; j < 20; ++j) {
                    histogram << counts_calcs[j] << ",";
                }
                histogram << endl;
                histogram.close();
            }

         
            
            search_duration = duration;
            search_dist_comp = hnsw->layer0_dist_comps;
            total_dist_comp = hnsw->layer0_dist_comps + hnsw->upper_dist_comps;
            candidates_popped = hnsw->candidates_popped;
            candidates_size = hnsw->candidates_size;
            candidates_without_if = hnsw->candidates_without_if;

            if (neighbors.empty())
                break;

            cout << "Results for construction parameters: " << config->optimal_connections << ", " << config->max_connections << ", "
                << config->max_connections_0 << ", " << config->ef_construction << " and search parameters: " << config->ef_search << endl;
            
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
        }

        // Update benchmark file statistics
        double recall = (double) similar / (config->num_queries * config->num_return);
        cout << "Correctly found neighbors: " << similar << " ("
             << recall * 100 << "%)" << endl;
        double average_ndcg = (double) total_ndcg / config->num_queries;
        cout << "Average NDCG@" << config->num_return << ": " << average_ndcg << endl;
        if (config->export_benchmark) {
            std::string dist_comp_layer0_string = config->export_median_calcs ? std::to_string(median_comps_layer0) : std::to_string(search_dist_comp / config->num_queries);
            std::string line = std::to_string(parameter) + ", " 
                     + dist_comp_layer0_string + ", "
                     + std::to_string(total_dist_comp / config->num_queries) + ", " 
                     + std::to_string(recall) + ", " 
                     + std::to_string(search_duration / config->num_queries) + ", "
                     + std::to_string(average_ndcg) + ", "
                     + std::to_string(candidates_popped / config->num_queries) + ", "
                     + std::to_string(candidates_size / static_cast<float>(candidates_without_if)) + ", ";
            if (config->export_clustering_coefficient) {
                line += std::to_string(hnsw->calculate_average_clustering_coefficient()) + ", ";
                line += std::to_string(hnsw->calculate_global_clustering_coefficient()) + ", ";
            }
            if (config->use_hybrid_termination) {
                float estimated_distance_calcs = config->bw_slope != 0 ? (config->ef_search - config->bw_intercept) / config->bw_slope : 1;
                float termination_alpha = config->use_distance_termination ? config->termination_alpha : config->alpha_coefficient * log(estimated_distance_calcs) + config->alpha_intercept;
                line += std::to_string(hnsw->num_distance_termination ) + "---" + std::to_string(hnsw->num_original_termination) + ", ";
                line += std::to_string(termination_alpha) + ", ";
            }
            lines.push_back(line);
        }
        
        // Conditionally save graph
        if (config->export_graph && !config->load_graph_file) {
            hnsw->to_files(config, parameter_name + "_" + to_string(parameter_values[i]), construction_duration);
        }

        delete hnsw;
    }
    // Write to benchmark file
    if (config->export_benchmark) {
        if( config->export_median_calcs)
            *results_file << "note that distance at layer 0 here is median" << endl;
        *results_file << "\nparameter, dist_comps/query, total_dist_comp/query, recall, runtime/query, Avg NDCG, alpha, ratio termination (distance based/original), candidates_popped/query, candidates_size/candidates_without_if" << endl;
        for(auto& line: lines)
            *results_file << line <<endl;
        *results_file << endl << endl;
    }
    parameter = default_parameter;
    if (results_file != NULL) {
        results_file->close();
        delete results_file;
        cout << "Results exported to " << config->runs_prefix << "benchmark.txt" << endl;
    }

    if(config->export_candidate_popping_times){
        HNSW* hnsw = new HNSW(config, nodes);
        ofstream histogram = ofstream(config->metric_prefix + "_histogram_popping_times.txt");
        histogram <<  hnsw->candidate_popping_times.size() <<endl ;
        
        int count = 0;
        for (auto& l:  hnsw->candidate_popping_times) {
            for(int j = 0; j < l.second.size(); j++){
                histogram << l.second[j]<< ",";
            }
             histogram << "\n";
            count++;
        }
        delete hnsw;


    }


}

void run_benchmarks(Config* config, float** nodes, float** queries, float** training) {
    // Initialize output files
    ofstream* results_file = NULL;
    if (config->export_benchmark) {
        results_file = new ofstream(config->runs_prefix + "benchmark.txt");
        *results_file << "Size " << config->num_nodes << "\nBenchmarking with Parameters: opt_con = "
                << config->optimal_connections << ", max_con = " << config->max_connections << ", max_con_0 = " << config->max_connections_0
                << ", ef_con = " << config->ef_construction << ", scaling_factor = " << config->scaling_factor
                << ", ef_search = " << config->ef_search 
                
                << "\nnum_return = " << config->num_return << ", learning_rate = " << config->learning_rate << ", initial_temperature = " << config->initial_temperature
                << ", decay_factor = " << config->decay_factor << ", initial_keep_ratio = " << config->initial_keep_ratio
                << ", final_keep_ratio = " << config->final_keep_ratio << ", grasp_loops = " << config->grasp_loops  << ", Single training = " << config->single_ep_training 
                
                <<"\nCurrent Run Properties: Stinky Values = "  << std::boolalpha  <<  config->use_stinky_points << " [" <<config->stinky_value <<"]" 
                << ", use_heuristic = " << config->use_heuristic << ", use_grasp = " << config->use_grasp << ", use_dynamic_sampling = " << config->use_dynamic_sampling  << ", use_cost_benefit = " << config->use_cost_benefit 
                << ", use_direct_path = " << config->use_direct_path << ", Single construction point = " << config->single_ep_construction  << ", Single Search Point =  " << config->single_ep_query  << ", ef_search_upper = " << config->ef_search_upper << ", k_upper = " << config->k_upper
                << ", current Pruning method = " << config->weight_selection_method 
                << "\nCurrent Termination Method : "
                << "\nUse_distance_termination = " << config->use_distance_termination  << ", use_hybrid_termination " << config->use_hybrid_termination <<  ", use_latest: " << config->use_latest  << ", use_median: " << config->use_median_equations
                << ", Use Break = " << config->use_break << ", Use median break = " << config->use_median_break <<", Use median earlist = " << config->use_median_earliast << ", use_calculation_termination = " << config->use_calculation_termination  << ", use oracle 1 = "  << config->use_groundtruth_termination << ", use_calculation_oracle = " << config->use_calculation_oracle 
                << "\nTermination values : "
                << "Distance Termination alpha = " << config->alpha_termination_selection  << ", alpha break = " << config->alpha_break << ", efs break = " << config->efs_break  <<  ", Median Break value = " << config->breakMedian  << endl;


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
        if (config->export_calcs_per_query) {
            ofstream histogram = ofstream(config->runs_prefix + "histogram_calcs_per_query.txt");
            histogram.close();
        }
        if(config->export_candidate_popping_times){
            ofstream histogram = ofstream(config->metric_prefix + "_histogram_popping_times.txt");
            histogram.close();

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
    run_benchmark(config, config->termination_alpha, config->benchmark_termination_alpha, "termination_alpha", nodes,
        queries, training, results_file);
    run_benchmark(config, config->ef_search, config->benchmark_ef_search, "ef_search", nodes,
        queries, training, results_file);
    run_benchmark(config, config->calculations_per_query, config->benchmark_calculations_per_query, "calculations_per_query", nodes,
        queries, training, results_file);
    run_benchmark(config, config->oracle_termination_total, config->benchmark_oracle_termination_total, "oracle_termination_total", nodes,
        queries, training, results_file);

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
    }
}


/**
 * This class is used to run HNSW with different parameters, comparing the recall
 * versus ideal for each set of parameters.
*/
int main() {
    // string CPUBrand = get_cpu_brand();
    time_t now = time(0);
    cout << "Benchmark run started at " << ctime(&now);
    Config* config = new Config();

    // Load nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** queries = new float*[config->num_queries];
    load_queries(config, nodes, queries);
    float** training = nullptr;
    // if ((config->use_grasp || config->use_cost_benefit) && !config->load_graph_file) {
    //     training = new float*[config->num_training];
    //     load_training(config, nodes, training, config->num_training);
    //     remove_duplicates(config, training, queries, config->num_queries);
    // }

    // Run benchmarks for each set of grid parameters
    int grid_size = min({config->grid_num_return.size(), config->grid_runs_prefix.size(), config->grid_graph_file.size()});
    if (grid_size == 0) {
        run_benchmarks(config, nodes, queries, training);
    } else {
        for (int i = 0; i < grid_size; ++i) {
            int default_num_return = config->num_return;
            string default_runs_prefix = config->runs_prefix;
            string default_graph_file = config->loaded_graph_file;
            string default_info_file = config->loaded_info_file;
            config->num_return = config->grid_num_return[i];
            config->runs_prefix = config->grid_runs_prefix[i];
            config->loaded_graph_file = config->grid_graph_file[i];
            config->loaded_info_file = std::regex_replace(std::regex_replace(config->grid_graph_file[i], std::regex("graph"), "info"), std::regex("bin"), "txt");
            run_benchmarks(config, nodes, queries, training);
            config->num_return = default_num_return;
            config->runs_prefix = default_runs_prefix;
            config->loaded_graph_file = default_graph_file;
            config->loaded_info_file = default_info_file;
        }
    }

    // Clean up
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
