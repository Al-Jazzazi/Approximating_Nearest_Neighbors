#include <iostream>
#include <vector>
#include "grasp.h"
#include <math.h>
#include <utility>
#include <random>
#include <cfloat> 
#include <algorithm>
#include <iomanip>

#include "../HNSW/hnsw.h"

using namespace std;

/**
 * Alg 1
 * Given an HNSW, a list of its weighted edges, and a list of training nodes,
 * learn the importance of the HNSW's edges and increase their weights accordingly.
 * Note: This will shuffle the training set.
 */
void learn_edge_importance(Config* config, HNSW* hnsw, vector<Edge*>& edges, float** training) {
    float temperature = config->initial_temperature;
    float lambda = 0;
    mt19937 gen(config->graph_seed);

    // Run the training loop
    for (int k = 0; k < config->grasp_loops; k++) {
        for (int j = 0; j < config->grasp_subloops; j++) {
            lambda = compute_lambda(config->final_keep_ratio, config->initial_keep_ratio, k, config->grasp_loops, config->keep_exponent);
            if (j == config->grasp_subloops - 1) {
                normalize_weights(config, hnsw, edges, lambda, temperature);
            }
            if (!config->use_dynamic_sampling) {
                sample_subgraph(config, edges, lambda);
            }
            update_weights(config, hnsw, training, config->num_return);
            temperature = config->initial_temperature * pow(config->decay_factor, k);
            std::shuffle(training, training + config->num_training, gen);
            // cout << "Temperature: " << temperature << " Lambda: " << lambda << endl;
        }
    }
}

/**
 * Alg 2
 * Normalize the edge weights in the HNSW according to a normalization factor,
 * which is computed from the weight range, lambda, and temperature
 */
void normalize_weights(Config* config, HNSW* hnsw, vector<Edge*>& edges, float lambda, float temperature) {
    // Compute normalizing factor mu
    float target = lambda * edges.size();
    pair<float,float> max_min = find_max_min(config, hnsw);
    float avg_w = temperature * log(lambda / (1 - lambda));
    float search_range_min = avg_w - max_min.first;
    float search_range_max = avg_w - max_min.second;
    float mu = binary_search(config, edges, search_range_min, search_range_max, target, temperature);
    // cout << "Mu: " << mu << " Min: " << max_min.second << " Max: " << max_min.first << " Avg: " << avg_w << endl;
    
    int* counts = new int[20];
    for (int i = 0; i < 20; i++) {
        counts[i] = 0;
    }

    // Normalize edge weights and probabilities
    for(int i = 0; i < config->num_nodes ; i++){
        for(int k = 0; k < hnsw->mappings[i][0].size(); k++){
            Edge& edge = hnsw->mappings[i][0][k];
            int count_position = edge.probability_edge == 1 ? 19 : edge.probability_edge * 20;
            edge.weight += mu;
            edge.probability_edge = 1 / (1 + exp(-edge.weight / temperature));
            counts[count_position]++;
        }
    }
    // Record probability distribution in histogram text file
    if (!config->histogram_prob_file.empty()) {
        ofstream histogram = ofstream(config->histogram_prob_file, std::ios::app);
        for (int i = 0; i < 20; i++) {
            histogram << counts[i] << ",";
        }
        histogram << endl;
        histogram.close();
    }
    delete[] counts;
}

/**
 * Given an HNSW and a list of its edges, keep its num_keep highest weighted
 * edges and remove the rest of its edges.
 */
void prune_edges(Config* config, HNSW* hnsw, vector<Edge*>& edges, int num_keep) {
    // Mark lowest weight edges for deletion
    auto compare = [](Edge* lhs, Edge* rhs) { return lhs->probability_edge > rhs->probability_edge; };
    priority_queue<Edge*, vector<Edge*>, decltype(compare)> remaining_edges(compare);
    for (int i = 0; i < edges.size(); i++) {
        // Enable edge by default
        edges[i]->ignore = false;
        remaining_edges.push(edges[i]);
        // Disable edge if it is pushed out of remaining edges
        if (remaining_edges.size() > num_keep) {
            remaining_edges.top()->ignore = true;
            remaining_edges.pop();
        }
    }
    // Remove all edges in layer 0 that are marked for deletion
    for (int i = 0; i < hnsw->num_nodes; i++) {
        for (int j = hnsw->mappings[i][0].size() - 1; j >= 0; j--) {
            vector<Edge>& neighbors = hnsw->mappings[i][0];
            if (neighbors[j].ignore) {
                neighbors[j] = neighbors[neighbors.size() - 1];
                neighbors.pop_back();
            }
        }
    }
}

/**
 * Compare the nearest neighbors and paths taken on the sampled graph with
 * the original graph, and increase edge weights accordingly
 */
void update_weights(Config* config, HNSW* hnsw, float** training, int num_neighbors) {
    int num_updates = 0;
    int num_of_edges_updated = 0;
    for (int i = 0; i < config->num_training; i++) {
        int similar_nodes = 0;

        // Find the nearest neighbor and paths taken using the original and sampled graphs
        pair<int, float*> query = make_pair(i, training[i]);
        vector<vector<Edge*>> sample_path;
        vector<vector<Edge*>> original_path;
        vector<pair<float, int>> sample_nearest = hnsw->nn_search(config, sample_path, query, num_neighbors, true);
        vector<pair<float, int>> original_nearest = hnsw->nn_search(config, original_path, query, num_neighbors, false);

        // Calculate the average distance between nearest neighbors and the training point
        float sample_distance = 0;
        float original_distance = 0;
        for (int j = 0; j < num_neighbors; j++) {
            sample_distance += sample_nearest[j].first;
            original_distance += original_nearest[j].first;
        }

             for (int j = 0; j < sample_path[0].size(); j++) 
                 sample_path[0][j]->weight += config->stinkyValue;

            for (int j = 0; j < original_path[0].size(); j++) {
                 original_path[0][j]->weight += config->stinkyValue;

                if ((config->weight_selection_method == 0) ||
                    (config->weight_selection_method == 1 && original_path[0][j]->ignore) ||
                    (config->weight_selection_method == 2 && find(sample_path[0].begin(), sample_path[0].end(), original_path[0][j]) == sample_path[0].end())
                ) {
                    original_path[0][j]->weight += (sample_distance / original_distance - 1) * config->learning_rate;
                    num_of_edges_updated++;
                }
    
            num_updates++;
        }
        
    }
    if (config->print_weight_updates) {
        cout << "# of Weight Updates: " << num_updates << " / " << config->num_training << ", # of Edges Updated: " << num_of_edges_updated << endl; 
    }
}

/**
 * Randomly disable edges in the provided list of edges
 */
void sample_subgraph(Config* config, vector<Edge*>& edges, float lambda) {
    //mark any edge less than a randomly created probability as ignored, thus creating a subgraph with less edges 
    //Note: the number is not necessarily lambda * E 
    mt19937 gen(config->graph_seed);
    uniform_real_distribution<float> dis(0, lambda);
    int count = 0;
    for(Edge* edge : edges) {
        if (dis(gen) < (1 - edge->probability_edge)) {
            edge->ignore = true;
            count++;
        } else {
            edge->ignore = false; 
        }
        
    }

   // cout << "Number of edges ignored: " << count << endl;

}

// Calculate lambda according to a formula
float compute_lambda(float final_keep, float initial_keep, int k, int num_iterations, int c) {
    return final_keep + (initial_keep - final_keep) * pow(1 - static_cast<float>(k) / num_iterations, c);
}
 
// Find the maximum and minimum weights in the HNSW
pair<float,float> find_max_min(Config* config, HNSW* hnsw) {
    float max_w = 0.0f; 
    float min_w = FLT_MAX; 
    float lowest_percentage = 1.0f;
    float max_probability = 0.0f;
    pair<float,float> max_min;
    for(int i = 0; i < config->num_nodes ; i++){
        for(int k = 0; k < hnsw->mappings[i][0].size(); k++){
            if(max_w < hnsw->mappings[i][0][k].weight)
                max_w = hnsw->mappings[i][0][k].weight;
            
            if(min_w > hnsw->mappings[i][0][k].weight)
                min_w = hnsw->mappings[i][0][k].weight;

            //Used to check edge probability behaviour at each iteration
            // if (lowest_percentage > hnsw->mappings[i][0][k].probability_edge)
            //     lowest_percentage = hnsw->mappings[i][0][k].probability_edge;

            // if(max_probability < hnsw->mappings[i][0][k].probability_edge)
            //     max_probability = hnsw->mappings[i][0][k].probability_edge;
        }
    }
    //cout << "lowest prob is :" << lowest_percentage <<  " Max prob is: " <<  max_probability << endl;
    max_min = make_pair(max_w, min_w);
    return max_min;
}

/**
 * Binary search for the mu value that makes the sum of edge probabilities
 * equal lambda * mu
 */
float binary_search(Config* config, vector<Edge*>& edges, float left, float right, float target, float temperature) {
    const double EPSILON = 1e-3; // Tolerance for convergence
    float sum_of_probabilities = 0;
     

    // The function keeps updating value of mu -mid in this case- to recalculating the probabilities such that 
    // sum of probabilites gets as close as lambda * E.
    while (right - left > EPSILON) {
        float mid = left + (right - left) / 2;
        for (const Edge* edge : edges) {
            sum_of_probabilities += 1/(1 + exp(-(edge->weight + mid) / temperature));
        }
        if(abs(sum_of_probabilities - target) < 1.0f)
            break;
        else if (sum_of_probabilities < target) 
            left = mid; 
         else 
            right = mid; 
        sum_of_probabilities = 0;

        //std::cout << std::setprecision(11);
       //cout << "left: " << left << " Right " << right << " MID" << mid << endl;
    }

    return left + (right - left) / 2;
}

// Load training set from training file or randomly generate them from nodes
void load_training(Config* config, float** nodes, float** training) {
    mt19937 gen(config->training_seed);
    if (config->training_file != "") {
        if (config->query_file.size() >= 6 && config->training_file.substr(config->training_file.size() - 6) == ".fvecs") {
            // Load training from fvecs file
            load_fvecs(config->training_file, "training", training, config->num_training, config->dimensions, config->groundtruth_file != "");
            return;
        }

        // Load training from file
        ifstream f(config->training_file, ios::in);
        if (!f) {
            cout << "File " << config->training_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << config->num_training << " training set from file " << config->training_file << endl;

        for (int i = 0; i < config->num_training; i++) {
            training[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> training[i][j];
            }
        }

        f.close();
        return;
    }

    if (config->load_file == "") {
        // Generate random training nodes
        cout << "Generating " << config->num_training << " random training points" << endl;
        uniform_real_distribution<float> dis(config->gen_min, config->gen_max);

        for (int i = 0; i < config->num_training; i++) {
            training[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                training[i][j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
            }
        }

        return;
    }
    
    // Generate training set randomly based on bounds of graph_nodes
    cout << "Generating training set based on file " << config->load_file << endl;
    float* lower_bound = new float[config->dimensions];
    float* upper_bound = new float[config->dimensions];
    std::copy(nodes[0], nodes[0] + config->dimensions, lower_bound);
    std::copy(nodes[0], nodes[0] + config->dimensions, upper_bound);

    // Calculate lowest and highest value for each dimension using graph_nodes
    for (int i = 1; i < config->num_nodes; i++) {
        for (int j = 0; j < config->dimensions; j++) {
            if (nodes[i][j] < lower_bound[j]) {
                lower_bound[j] = nodes[i][j];
            }
            if (nodes[i][j] > upper_bound[j]) {
                upper_bound[j] = nodes[i][j];
            }
        }
    }
    uniform_real_distribution<float>* dis_array = new uniform_real_distribution<float>[config->dimensions];
    for (int i = 0; i < config->dimensions; i++) {
        dis_array[i] = uniform_real_distribution<float>(lower_bound[i], upper_bound[i]);
    }

    // Generate training set based on the range of values in each dimension
    for (int i = 0; i < config->num_training; i++) {
        training[i] = new float[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            training[i][j] = round(dis_array[j](gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
    }

    delete[] lower_bound;
    delete[] upper_bound;
    delete[] dis_array;
}

// Remove training points that are also found in queries
void remove_duplicates(Config* config, float** training, float** queries) {
    int num_training_filtered = config->num_training;
    for (int i = config->num_training - 1; i >= 0; i--) {
        for (int j = 0; j < config->num_queries; j++) {
            bool is_same = true;
            for (int d = 0; d < config->dimensions; d++) {
                if (training[i][d] != queries[j][d]) {
                    is_same = false;
                    break;
                }
            }
            if (is_same) {
                delete[] training[i];
                training[i] = training[num_training_filtered - 1];
                training[num_training_filtered - 1] = nullptr;
                num_training_filtered--;
                break;
            }
        }
    }
    config->num_training = num_training_filtered;
}
