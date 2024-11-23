#include <fstream>
#include <fstream>
#include <chrono> 
#include "graph.h"
#include "utils.h"
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

        vector<vector<int>> neighbors;
        double construction_duration, search_duration;
        long long search_dist_comp, total_dist_comp, candidates_popped, candidates_size, candidates_without_if ;
        Graph G(config);
        G.load(config);

        vector<vector<int>> actual_neighbors;

        get_actual_neighbors(config, actual_neighbors, nodes, queries);

        int startNode = G.start;

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

        G.reset_statistics();

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
            vector<long long int> dist_comps_per_q_vec;

            if(config->use_hybrid_termination || config->use_distance_termination ) 
                G.calculate_termination(config);
        

            for (int j = 0; j < config->num_queries; ++j) {

             try {
                if (j >= actual_neighbors.size()) 
                    throw std::out_of_range("Index out of bounds in actual_neighbors");

                G.cur_groundtruth = actual_neighbors[j];
                } 
                catch (const std::out_of_range& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                }

                vector<int> result;
                BeamSearch(G,config, startNode, queries[j], config->ef_search, result);
                neighbors.emplace_back(result);
                // neighbors.emplace_back(hnsw->nn_search(config, path, query_pair, config->num_return));

                if (config->export_calcs_per_query) {
                    ++counts_calcs[std::min(19, (int)G.distanceCalculationCount / config->interval_for_calcs_histogram)];
                }
                if (config->export_median_calcs || config->export_median_precentiles) {
                    dist_comps_per_q_vec.push_back(G.distanceCalculationCount);
                }

            }



            // Log search statistics
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            cout << "Query time: " << duration / 1000.0 << " seconds, ";
            cout << "Distance computations (layer 0): " << G.distanceCalculationCount << ", ";

            if (config->export_median_calcs) {
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
            if (config->export_calcs_per_query) {
                ofstream histogram = ofstream(config->runs_prefix + "histogram_calcs_per_query.txt", std::ios::app);
                for (int j = 0; j < 20; ++j) {
                    histogram << counts_calcs[j] << ",";
                }
                histogram << endl;
                histogram.close();
            }

         
            
            search_duration = duration;
            search_dist_comp = G.distanceCalculationCount;
            total_dist_comp = search_dist_comp;
            // candidates_popped = G.candidates_popped;
            // candidates_size = G.candidates_size;
            // candidates_without_if = G.candidates_without_if;

            if (neighbors.empty())
                break;

            cout << "Results for construction parameters: " << config->optimal_connections << ", " << config->max_connections << ", "
                << config->max_connections_0 << ", " << config->ef_construction << " and search parameters: " << config->ef_search << endl;
            
            find_similar(config, actual_neighbors, neighbors, nodes, queries, similar, total_ndcg);
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
                    //  + std::to_string(total_dist_comp / config->num_queries) + ", " 
                     + std::to_string(recall) + ", " 
                     + std::to_string(G.num_set_checks/ config->num_queries) + ", "
                     + std::to_string(G.size_of_c / config->num_queries) + ", "
                     + std::to_string(G.num_insertion_to_c / config->num_queries) + ", "
                     + std::to_string(G.num_deletion_from_c / config->num_queries) + ", "
                     + std::to_string(G.size_of_visited / config->num_queries) + ", "           
                     + std::to_string(search_duration / config->num_queries) + ", ";
                    
                   //  + std::to_string(candidates_popped / config->num_queries) + ", "
                     //+ std::to_string(candidates_size / static_cast<float>(candidates_without_if)) + ", ";
       
            if (config->use_hybrid_termination) {
                float estimated_distance_calcs = config->bw_slope != 0 ? (config->ef_search - config->bw_intercept) / config->bw_slope : 1;
                float termination_alpha = config->use_distance_termination ? config->termination_alpha : config->alpha_coefficient * log(estimated_distance_calcs) + config->alpha_intercept;
                line += std::to_string(G.num_distance_termination ) + "---" + std::to_string(G.num_original_termination) + ", ";
                line += std::to_string(termination_alpha) + ", ";

            }
            lines.push_back(line);
        }
        

    }
    // Write to benchmark file
    if (config->export_benchmark) {
        if( config->export_median_calcs)
            *results_file << "note that distance at layer 0 here is median" << endl;
        *results_file << "\nparameter, dist_comps/query, recall, num_set_checks/query,  size_of_c/query, num_insertion_to_c/query, num_deletion_from_c/query,size_of_visited/query,  runtime/query,  ratio termination (distance based/original), alpha" << endl;
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

   


}




void run_benchmarks(Config* config, float** nodes, float** queries, float** training) {
    // Initialize output files
    ofstream* results_file = NULL;
    if (config->export_benchmark) {
        results_file = new ofstream(config->runs_prefix + "benchmark.txt");
        *results_file << "Size " << config->num_nodes << "\nBenchmarking with Parameters:  L_construct = " << config->ef_construction << ", R = " << config->R 
                << ", alpha = " << config->alpha_vamana  << ", ef_search = " << config->ef_search << ", num_return = " << config->num_return 
                << "\nCurrent Termination Method : "
                << "\nUse_distance_termination = " << config->use_distance_termination  << ", use_hybrid_termination " << config->use_hybrid_termination <<  ", use_latest: " << config->use_latest  << ", use_median: " << config->use_median_equations
                << ", Use Break = " << config->use_break << ", Use median break = " << config->use_median_break << ", use_calculation_termination = " << config->use_calculation_termination  << ", use oracle 1 = "  << config->use_groundtruth_termination << ", use_calculation_oracle = " << config->use_calculation_oracle 
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
        if(config->export_median_precentiles){
            ofstream histogram = ofstream(config->metric_prefix + "_median_percentiles.txt");
            histogram.close();
        }
    }



    // run_benchmark(config, config->alpha_vamana, config->benchmark_alpha_vamana,
    //     "alpha_vamana", nodes, queries, training, results_file);
    // run_benchmark(config, config->ef_construction, config->benchmark_ef_construction,
    //     "R", nodes, queries, training, results_file);
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

   
}



//  * This class is used to run HNSW with different parameters, comparing the recall
//  * versus ideal for each set of parameters.
// */
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


