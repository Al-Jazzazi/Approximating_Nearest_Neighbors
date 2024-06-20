#include <iostream>
#include <vector>
#include "grasp.h"
#include <math.h>
#include <utility>
#include <random>
#include <cfloat> 
#include "../HNSW/hnsw.h"

using namespace std;

/**
 * Alg 1
 */
void learn_edge_importance(Config* config, HNSW* hnsw, vector<Edge*>& edges, float** nodes, float** queries) {
    float temperature = config->initial_temperature;
    float lambda = 0;
    mt19937 gen(config->graph_seed);

    for (int k = 0; k < config->grasp_iterations; k++) {
        int num_diff = 0;
        lambda = compute_lambda(config->final_keep_ratio, config->initial_keep_ratio, k, config->grasp_iterations, config->keep_exponent);
        normalize_weights(config, hnsw, edges, lambda, temperature);
        sample_subgraph(config, edges, lambda);

        for (int i = 0; i < config->num_training; i++) {
            // Find the nearest neighbor using both the original and sampled graphs
            pair<int, float*> query = make_pair(i, queries[i]);
            vector<vector<Edge*>> sample_path;
            vector<vector<Edge*>> original_path;
            vector<pair<float, int>> sample_nearest = hnsw->nn_search(config, sample_path, query, 1, true);
            vector<pair<float, int>> original_nearest = hnsw->nn_search(config, original_path, query, 1, false);
            
            // If the nearest neighbor differs, increase the weight importances
            if (original_nearest[0].second != sample_nearest[0].second) {
                float sample_distance = calculate_l2_sq(nodes[sample_nearest[0].second], queries[i], config->dimensions, 0);
                float original_distance = calculate_l2_sq(nodes[original_nearest[0].second], queries[i], config->dimensions, 0);
                num_diff++;
                if (original_distance != 0) {
                    for (int j = 0; j < original_path[0].size(); j++) {
                        original_path[0][j]->weight = original_path[0][j]->weight + (sample_distance / original_distance - 1) * config->learning_rate;
                    }
                }
            }
        }
        temperature = config->initial_temperature * pow(config->decay_factor, k);
        std::shuffle(queries, queries + config->num_training, gen);
        //cout << "Temperature: " << temperature << " Lambda: " << lambda << endl;
       // cout << "Num Different: " << num_diff << endl;
    }
}

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
 * Alg 2
 */
void normalize_weights(Config* config, HNSW* hnsw, vector<Edge*>& edges, float lambda, float temperature) {
    float target = lambda * edges.size();
    pair<float,float> max_min = find_max_min(config, hnsw);
    // float avg_w = (max_min.second  +  max_min.first)/2;
    float avg_w = temperature * log(lambda / (1 - lambda));

    float search_range_min = avg_w - max_min.first;
    float search_range_max = avg_w - max_min.second;

    float mu = binary_search(config, edges, search_range_min, search_range_max, target, temperature);
   // cout << "Mu: " << mu << " Min: " << max_min.second << " Max: " << max_min.first << " Avg: " << avg_w << endl;

    for(int i = 0; i < config->num_nodes ; i++){
        for(int k = 0; k < hnsw->mappings[i][0].size(); k++){
            Edge& edge = hnsw->mappings[i][0][k];
            edge.weight += mu;
            edge.probability_edge = 1 / (1 + exp(-edge.weight / temperature));
        }
    }

}

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

float compute_lambda(float final_keep, float initial_keep, int k, int num_iterations, int c) {
    return final_keep + (initial_keep - final_keep) * pow(1 - static_cast<float>(k) / num_iterations, c);
}
 
/**
 * Alg 2 helper Functions 
 */
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


            if (lowest_percentage > hnsw->mappings[i][0][k].probability_edge)
                lowest_percentage = hnsw->mappings[i][0][k].probability_edge;

            if(max_probability < hnsw->mappings[i][0][k].probability_edge)
                max_probability = hnsw->mappings[i][0][k].probability_edge;
        }
    }
    cout << "lowest prob is :" << lowest_percentage <<  " Max prob is: " <<  max_probability << endl;
    max_min = make_pair(max_w, min_w);
    return max_min;
}

float binary_search(Config* config, vector<Edge*>& edges, float left, float right, float target, float temperature) {
    const double EPSILON = 1e-6; // Tolerance for convergence
    float sum_of_probabilities = 0;
    // cout << "Range: " << left << " " << right << " Target: " << target;
    //The function keeps updating value of mu -mid in this case- to recalculating the probabilities such that 
    //sum of probabilites gets as close as lambda*E.
    while (right - left > EPSILON) {
        double mid = left + (right - left) / 2;
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
    }

    return left + (right - left) / 2;

}

// Load training set from training file or randomly generate them from nodes
void load_training(Config* config, float** nodes, float** training) {
    mt19937 gen(config->training_seed);
    if (config->training_file != "") {
        if (config->query_file.size() >= 6 && config->training_file.substr(config->training_file.size() - 6) == ".fvecs") {
            // Load queries from fvecs file
            load_fvecs(config->training_file, "training", training, config->num_training, config->dimensions, config->groundtruth_file != "");
            return;
        }

        // Load queries from file
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
        // Generate random queries (same as get_nodes)
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
    
    // Generate queries randomly based on bounds of graph_nodes
    cout << "Generating queries based on file " << config->load_file << endl;
    float* lower_bound = new float[config->dimensions];
    float* upper_bound = new float[config->dimensions];
    copy(nodes[0], nodes[0] + config->dimensions, lower_bound);
    copy(nodes[0], nodes[0] + config->dimensions, upper_bound);

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

    // Generate queries based on the range of values in each dimension
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