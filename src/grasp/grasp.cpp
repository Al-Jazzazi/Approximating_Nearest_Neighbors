#include <iostream>
#include <vector>
#include "grasp.h"
#include <math.h>
#include <utility>
#include <random>
#include <cfloat> 
#include <algorithm>
#include <iomanip>
#include <unordered_set>

#include "grasp.h"

using namespace std;

/* Scores each edge using cost-benefit points and prunes the 'config->final_keep_ratio'
 * edges with the lowest scores
 **/
void learn_cost_benefit(Config* config, HNSW* hnsw, vector<Edge*>& edges, float** training, int num_keep) {
    // Check how beneficial each edge is
    int total_benefit = 0;
    int total_cost = 0;
    int* total_cost_pointer = &total_cost;
    for (int i = 0; i < config->num_training; i++) {
        pair<int, float*> query = make_pair(i, training[i]);
        vector<Edge*> path;
        // Search for the query while counting the cost of each edge
        vector<pair<float, int>> nearest_neighbors = hnsw->nn_search(config, path, query, config->num_return, false, true, false, total_cost_pointer);
        for (int j = 0; j < path.size(); j++) {
            path[j]->benefit += 1;
            total_benefit += 1;
        }
    }
    
    // Initialize exports
    vector<long long> counts_cost;
    vector<long long> counts_benefit;
    for (int i = 0; i < 20; i++) {
        counts_cost.push_back(0);
        counts_benefit.push_back(0);
    }
    ofstream* pruned_file = nullptr;
    if (config->export_cost_benefit_pruned) {
        pruned_file = new ofstream(config->runs_prefix + "cost_benefit_pruned.txt");
    }

    // Compute average cost and benefit to use as a baseline for score comparisons
    float average_benefit = static_cast<float>(total_benefit) / edges.size();
    float average_cost = static_cast<float>(total_cost) / edges.size();
    auto compare = [average_benefit, average_cost](Edge* lhs, Edge* rhs) {
        return (average_benefit + lhs->benefit) / (average_cost + lhs->cost) > (average_benefit + rhs->benefit) / (average_cost + rhs->cost);
    };
    cout << "Average Benefit: " << average_benefit << " Average Cost: " << average_cost << endl;

    // Mark edges for deletion
    priority_queue<Edge*, vector<Edge*>, decltype(compare)> remaining_edges(compare);
    for (int i = 0; i < edges.size(); i++) {
        counts_cost[std::min(19, edges[i]->cost / config->interval_for_cost_histogram)]++;
        counts_benefit[std::min(19, edges[i]->benefit / config->interval_for_benefit_histogram)]++;
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
                if (config->export_cost_benefit_pruned) {
                    *pruned_file << neighbors[j].target << " " << neighbors[j].cost << " " << neighbors[j].benefit << endl;
                }
                neighbors[j] = neighbors[neighbors.size() - 1];
                neighbors.pop_back();
            }
        }
    }
    // Write and close exports
    if (config->export_histograms) {
        ofstream cost_histogram = ofstream(config->runs_prefix + "histogram_cost.txt", std::ios::app);
        ofstream benefit_histogram = ofstream(config->runs_prefix + "histogram_benefit.txt", std::ios::app);
        for (int i = 0; i < 20; i++) {
            cost_histogram << counts_cost[i] << ",";
            benefit_histogram << counts_benefit[i] << ",";
        }
        cost_histogram << endl;
        cost_histogram.close();
        benefit_histogram << endl;
        benefit_histogram.close();
    }
    if (config->export_cost_benefit_pruned) {
        pruned_file->close();
        delete pruned_file;
    }
}

/**
 * Alg 1
 * Given an HNSW, a list of its weighted edges, and a list of training nodes,
 * learn the importance of the HNSW's edges and increase their weights accordingly.
 * Note: This will shuffle the training set.
 */
void learn_edge_importance(Config* config, HNSW* hnsw, vector<Edge*>& edges, float** training, ofstream* results_file) {
    // Initialize parameters
    float temperature = config->initial_temperature;
    float lambda = 0;
    mt19937 gen(config->shuffle_seed);
    if (results_file != nullptr) {
        *results_file << "iteration\t# of Weights updated\t# of Edges updated\n"; 
    }

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
            if (results_file != nullptr) {
                *results_file << k;
            }
            update_weights(config, hnsw, training, config->num_return, results_file);

            temperature = config->initial_temperature * pow(config->decay_factor, k);
            std::shuffle(training, training + config->num_training, gen);
        }
        // Generate a new set of training sets each iteration
        if(config->generate_our_training && config->regenerate_each_iteration){
            load_training(config, hnsw->nodes, training, config->num_training);
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
    
    // Initialize edge distribution vectors
    int* counts_prob = new int[20];
    int* counts_w = new int [20];
    std::fill(counts_prob, counts_prob + 20, 0);
    std::fill(counts_w, counts_w + 20, 0);
  
    // Normalize edge weights and probabilities
    for(int i = 0; i < config->num_nodes ; i++){
        for(int k = 0; k < hnsw->mappings[i][0].size(); k++){
            Edge& edge = hnsw->mappings[i][0][k];
            int count_position = edge.probability_edge >= 1 ? 19 : edge.probability_edge * 20;
            edge.weight += mu;
            edge.probability_edge = 1 / (1 + exp(-edge.weight / temperature));
            counts_prob[count_position]++;
            // Update edge distributions
            if(edge.weight < 0) {
                counts_w[0]++;
            } else {
                count_position = edge.weight >=19*config->interval_for_weight_histogram ? 19: edge.weight/config->interval_for_weight_histogram +1; 
                counts_w[count_position]++;
            }
        }
    }
    // Record distributions in histogram text files
    if (config->export_histograms) {
        ofstream histogram = ofstream(config->runs_prefix + "histogram_prob.txt", std::ios::app);
        for (int i = 0; i < 20; i++) {
            histogram << counts_prob[i] << ",";
        }
        histogram << endl;
        histogram.close();

        histogram = ofstream(config->runs_prefix + "histogram_weights.txt", std::ios::app);
        for (int i = 0; i < 20; i++) {
            histogram << counts_w[i] << "," ;
        }
        histogram << "\t\t\t" << "Min W :" << max_min.second <<  " Max W is: " <<  max_min.first << endl;
        histogram.close();
    }
    delete[] counts_prob;
    delete[] counts_w;
}

/**
 * Given an HNSW and a list of its edges, keep its num_keep highest weighted
 * edges and remove the rest of its edges.
 */
void prune_edges(Config* config, HNSW* hnsw, vector<Edge*>& edges, int num_keep) {
    // Lower edge probabilities by stinky points
    if(config->use_stinky_points){
        for (auto e: edges){
            e->probability_edge -= config->stinky_value * e->stinky;
        }
    }
    // Mark lowest weight edges for deletion
    auto compare =[](Edge* lhs, Edge* rhs) { return lhs->probability_edge > rhs->probability_edge;};
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
void update_weights(Config* config, HNSW* hnsw, float** training, int num_neighbors, ofstream* results_file) {
    int num_updates = 0;
    int num_of_edges_updated = 0;
    for (int i = 0; i < config->num_training; i++) {
        int similar_nodes = 0;

        // Find the nearest neighbor and paths taken using the original and sampled graphs
        pair<int, float*> query = make_pair(i, training[i]);
        vector<Edge*> sample_path;
        vector<Edge*> original_path;
        vector<pair<float, int>> sample_nearest = hnsw->nn_search(config, sample_path, query, num_neighbors, false, true, true);
        vector<pair<float, int>> original_nearest = hnsw->nn_search(config, original_path, query, num_neighbors, false, true, false);
        unordered_set<Edge*> sample_path_set(sample_path.begin(), sample_path.end());
        double weight_change = calculate_weight_change(config, original_nearest, sample_nearest, results_file);

        // Add stinky points to each path edge
        if(config->use_stinky_points) {
            for (int j = 0; j < sample_path.size(); j++) 
                sample_path[j]->stinky += config->stinky_value;
            for (int j = 0; j < original_path.size(); j++)
                original_path[j]->stinky += config->stinky_value;
        }

        // Update edge weights if the change is non-zero
        if(weight_change != 0) {
            for (int j = 0; j < original_path.size(); j++) {
                // Select edges according to config->weight_selection_method
                if ((config->weight_selection_method == 0) ||
                    (config->weight_selection_method == 1 && original_path[j]->ignore) ||
                    (config->weight_selection_method == 2 && sample_path_set.find(original_path[j]) == sample_path_set.end())
                ) {
                    original_path[j]->weight += weight_change;
                    original_path[j]->num_of_updates++;
                    num_of_edges_updated++;
                }
            }
            num_updates++;
        }
    }

    // Create a histogram of the frequency of edge updates
    if(config->export_histograms){
        int* count_updates = new int [20];
        std::fill(count_updates, count_updates + 20, 0);
        for (int j = 0; j < config->num_nodes ; j++){
            for (int k = 0; k < hnsw->mappings[j][0].size(); k++){
                Edge& edge = hnsw->mappings[j][0][k];
                if (edge.num_of_updates == 0 ) {
                    count_updates[0]++;
                } else {
                    int count_position = edge.num_of_updates > 18*config->interval_for_num_of_updates_histogram ? 19 : edge.num_of_updates/config->interval_for_num_of_updates_histogram+1;
                    count_updates[count_position]++;
                }
            }
        }

        ofstream histogram = ofstream(config->runs_prefix + "histogram_edge_updates.txt", std::ios::app);
        for (int i = 0; i < 20; i++) {
            histogram << count_updates[i] << "," ;
        }
        histogram << endl; 
        histogram.close();
        delete[] count_updates;
    }
    if (config->print_weight_updates) {
        cout << "# of Weight Updates: " << num_updates << " / " << config->num_training << ", # of Edges Updated: " << num_of_edges_updated << endl; 
    }
    if (config->export_weight_updates && results_file != nullptr) {
        *results_file << "\t\t\t" <<num_updates << "\t\t\t\t" << num_of_edges_updated  <<endl; 
    }
    
}

double calculate_weight_change(Config* config, vector<pair<float, int>>& original_nearest, vector<pair<float, int>>& sample_nearest, ofstream* results_file) {
    double weight_change = 0;
    if (config->weight_formula == 0) {
        // Find the average distances between nearest neighbors and training point incrementally
        double sample_average = 0;
        double original_average = 0;
        for (int i = 0; i < sample_nearest.size(); i++) {
            sample_average += (sqrt(sample_nearest[i].first) - sample_average) / (i + 1);
        }
        for (int i = 0; i < original_nearest.size(); i++) {
            original_average += (sqrt(original_nearest[i].first) - original_average) / (i + 1);
        }
        // Calculate weight change from average distances
        if (original_average != 0) {
            weight_change = (sample_average / original_average - 1) * config->learning_rate;
        }
        if (config->export_negative_values && results_file != nullptr && weight_change < 0) {
            *results_file << "weight is being updated by a negative value" << endl;
        } 
    } else if (config->weight_formula == 1) {
        // Sum up the distance ratios for each pair of nearest neighbors
        double ratio_total = 0;
        for (int i = 0; i < original_nearest.size(); i++) {
            if (i >= sample_nearest.size()) {
                ratio_total += 100;
            } else if (original_nearest[i].first > 0) {
                ratio_total += sqrt(sample_nearest[i].first) / sqrt(original_nearest[i].first);
            }
        }
        // Calculate weight change from ratio total
        if (original_nearest.size() != 0) {
            weight_change = (ratio_total / original_nearest.size() - 1) * config->learning_rate;
        }
    } else if (config->weight_formula == 2) {
        // Convert sample_nearest into a set for efficient lookups
        unordered_set<int> sample_nearest_set;
        std::transform(sample_nearest.begin(), sample_nearest.end(), std::inserter(sample_nearest_set, sample_nearest_set.end()),
                      [](const pair<float, int>& neighbor) { return neighbor.second; });
        // Calculate weight change using discounted cumulative gain
        for (int i = 0; i < original_nearest.size(); i++) {
            if (sample_nearest_set.find(original_nearest[i].second) == sample_nearest_set.end()) {
                weight_change += 1 / log2(i + 2);
            }
        }
    }
    return weight_change;
}

/**
 * Randomly disable edges in the provided list of edges
 */
void sample_subgraph(Config* config, vector<Edge*>& edges, float lambda) {
    //mark any edge less than a randomly created probability as ignored, thus creating a subgraph with less edges 
    //Note: the number is not necessarily lambda * E 
    mt19937 gen(config->sample_seed);
    normal_distribution<float> dis(0, lambda);
    int count = 0;
    for(Edge* edge : edges) {
        if (dis(gen) < (1 - edge->probability_edge)) {
            edge->ignore = true;
            count++;
        } else {
            edge->ignore = false; 
        }
        
    }
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
        }
    }
    if (config->print_weight_updates) {
        cout << "Min W :" << min_w <<  " Max W is: " <<  max_w << endl;
    }
    max_min = make_pair(max_w, min_w);
    return max_min;
}

/**
 * Binary search for the mu value (mid) that makes the sum of probabilities
 * equal lambda * E.
 */
float binary_search(Config* config, vector<Edge*>& edges, float left, float right, float target, float temperature) {
    float sum_of_probabilities = 0;
    int count = 0;
    // Stops when the difference between endpoints is less than the specified precision
    // or when the number of iterations reaches the specified limit
    while ((right - left > 1e-3) && count < 1000) {
        count++;
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
    }

    return left + (right - left) / 2;
}

// Load training set from training file or randomly generate them from nodes
void load_training(Config* config, float** nodes, float** training, int num_training, bool is_generating) {
    std::random_device rd;
    mt19937 gen(rd());
   
    if (!is_generating && config->training_file != "") {
        if (config->query_file.size() >= 6 && config->training_file.substr(config->training_file.size() - 6) == ".fvecs") {
            // Load training from fvecs file
            load_fvecs(config->training_file, training, num_training, config->dimensions);
            return;
        }

        // Load training from file
        ifstream f(config->training_file, ios::in);
        if (!f) {
            cout << "File " << config->training_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << num_training << " training set from file " << config->training_file << endl;

        for (int i = 0; i < num_training; i++) {
            training[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> training[i][j];
            }
        }

        f.close();
        return;
    }

    if (!is_generating && config->load_file == "") {
        // Generate random training nodes
        cout << "Generating " << num_training << " random training points" << endl;
        normal_distribution<float> dis(config->gen_min, config->gen_max);

        for (int i = 0; i < num_training; i++) {
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
    normal_distribution<float>* dis_array = new normal_distribution<float>[config->dimensions];
    for (int i = 0; i < config->dimensions; i++) {
        dis_array[i] = normal_distribution<float>(lower_bound[i], upper_bound[i]);
    }

    // Generate training set based on the range of values in each dimension
    for (int i = 0; i < num_training; i++) {
        training[i] = new float[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            training[i][j] = round(dis_array[j](gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
    }

    delete[] lower_bound;
    delete[] upper_bound;
    delete[] dis_array;

}

// Remove training points that are also found in the other array
void remove_duplicates(Config* config, float** training, float** other, int other_num) {
    int num_training_filtered = config->num_training;
    for (int i = config->num_training - 1; i >= 0; i--) {
        for (int j = 0; j < other_num; j++) {
            bool is_same = true;
            for (int d = 0; d < config->dimensions; d++) {
                if (training[i][d] != other[j][d]) {
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
