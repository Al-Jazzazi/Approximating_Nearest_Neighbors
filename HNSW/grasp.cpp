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

void learn_cost_benefit(Config* config, HNSW* hnsw, vector<Edge*>& edges, float** training, int num_keep) {
    // Check how beneficial each edge is
    for (int i = 0; i < config->num_training; i++) {
        pair<int, float*> query = make_pair(i, training[i]);
        vector<Edge*> path;
        vector<pair<float, int>> nearest_neighbors = hnsw->nn_search(config, path, query, config->num_return, false, false, true);
        for (int j = 0; j < path.size(); j++) {
            path[j]->benefit++;
        }
    }
    // Print averages
    float total_cost = 0;
    float total_benefit = 0;
    for (int i = 0; i < edges.size(); i++) {
        total_cost += edges[i]->cost;
        total_benefit += edges[i]->benefit;
    }
    cout << "Cost: " << (total_cost / edges.size()) << " Benefit: " << (total_benefit / edges.size()) << endl;
    // Mark edges for deletion
    auto compare = [](Edge* lhs, Edge* rhs) {
        if (lhs->cost == 0) {
            return true;
        } else if (rhs->cost == 0) {
            return false;
        } else {
            return static_cast<float>(lhs->benefit) / lhs->cost > static_cast<float>(rhs->benefit) / rhs->cost;
        }
    };
    // auto compare = [](Edge* lhs, Edge* rhs) { return lhs->benefit - 0.5 * lhs->cost > rhs->benefit - 0.5 * rhs->cost; };
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
    int count = 0;
    for (int i = 0; i < hnsw->num_nodes; i++) {
        for (int j = hnsw->mappings[i][0].size() - 1; j >= 0; j--) {
            vector<Edge>& neighbors = hnsw->mappings[i][0];
            if (neighbors[j].ignore) {
                neighbors[j] = neighbors[neighbors.size() - 1];
                neighbors.pop_back();
                count++;
            }
        }
    }
    // Print averages
    total_cost = 0;
    total_benefit = 0;
    for (int i = 0; i < edges.size(); i++) {
        total_cost += edges[i]->cost;
        total_benefit += edges[i]->benefit;
    }
    cout << "Cost: " << (total_cost / edges.size()) << " Benefit: " << (total_benefit / edges.size()) << " Removed: " << count << endl;
}

/**
 * Alg 1
 * Given an HNSW, a list of its weighted edges, and a list of training nodes,
 * learn the importance of the HNSW's edges and increase their weights accordingly.
 * Note: This will shuffle the training set.
 */
void learn_edge_importance(Config* config, HNSW* hnsw, vector<Edge*>& edges, float** training, ofstream* results_file) {
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
            int num_return = config->num_return_training == -1 ? config->num_return : config->num_return_training;
            if (results_file != nullptr) {
                *results_file << k;
            }
            update_weights(config, hnsw, training, num_return, results_file);

            temperature = config->initial_temperature * pow(config->decay_factor, k);
            std::shuffle(training, training + config->num_training, gen);
            // cout << "Temperature: " << temperature << " Lambda: " << lambda << endl;
        }
        //Each loop, generate a new set training sets
        if(config->generate_our_training && config->regenerate_each_iteration){
            load_training(config, hnsw->nodes, training );
            //cout << "training" << training[10][10] << endl;
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
    
    int* counts_prob = new int[20];
    int* counts_w = new int [20];
    std::fill(counts_prob, counts_prob + 20, 0);
    std::fill(counts_w, counts_w + 20, 0);
  
    // Normalize edge weights and probabilities
    for(int i = 0; i < config->num_nodes ; i++){
        for(int k = 0; k < hnsw->mappings[i][0].size(); k++){
            Edge& edge = hnsw->mappings[i][0][k];
            int count_position = edge.probability_edge == 1 ? 19 : edge.probability_edge * 20;
            edge.weight += mu;
            edge.probability_edge = 1 / (1 + exp(-edge.weight / temperature));
            counts_prob[count_position]++;

            if(edge.weight < 0)
                counts_w[0]++;
            else{
            count_position = edge.weight >=19*config->interval_for_weight_histogram ? 19: edge.weight/config->interval_for_weight_histogram +1; 
            counts_w[count_position]++;
            }
        }
    }
    // Record distributions in histogram text files
    if (!config->runs_prefix.empty()) {
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
    //update probs 
    if(config->use_stinky_points ){
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
        vector<pair<float, int>> sample_nearest = hnsw->nn_search(config, sample_path, query, num_neighbors, true, config->use_stinky_points);
        vector<pair<float, int>> original_nearest = hnsw->nn_search(config, original_path, query, num_neighbors, false, config->use_stinky_points);
        unordered_set<Edge*> sample_path_set(sample_path.begin(), sample_path.end());

        vector<Edge*> direct_path; 
        int size = 0; 
        for(auto edge: sample_path){
            if(edge->target == sample_nearest[0].second){
                direct_path.push_back(edge);
                cout << "Edge point at closest element was found " << endl;  
                break;
            }
        }

        // while(size < sample_path.size() && !direct_path.empty() && direct_path.back()->prev_edge != nullptr){
        //     direct_path.push_back(direct_path.back()->prev_edge);
        //     if(direct_path.back()->prev_edge == nullptr){
        //        // cout << "We have successfully reached the start edge" << endl;
        //         // if(!direct_path.empty())
        //         //     direct_path.pop_back();
        //         // else 
        //         //     cerr << "direct path is empty" << endl;
        //         break;
        //     }
        //     size++;
        // }
        if(direct_path.empty()){
            cerr << "start edge wasn't found " << endl;
        }
            

        // Calculate the average distances between nearest neighbors and training point incrementally
        double sample_average = 0;
        double original_average = 0;
        for (int j = 0; j < num_neighbors; j++) {
            sample_average += (sample_nearest[j].first - sample_average) / (j + 1);
            original_average += (original_nearest[j].first - original_average) / (j + 1);
        }

        double num = (sample_average / original_average - 1) * config->learning_rate;
        if( config->export_negative_values && num < 0){
           if(results_file != nullptr)
                *results_file << "error weight is being updates by a negative value" << endl;
            else
                cout << "negative value found, num is " << num <<  ", sample distance is " << sample_average 
                << ", original distance is " << original_average << ", ration is " << sample_average / original_average << endl; 

        } 
        
        if(config->use_stinky_points){
            for (int j = 0; j < sample_path.size(); j++) 
                sample_path[j]->stinky += config->stinky_value;
            for (int j = 0; j < original_path.size(); j++)
                original_path[j]->stinky += config->stinky_value;
        }



        //Based on what we select to be the value of weight_selection_methon in config, the edges selected to be updated 
        //will differ 
        if(sample_average != original_average) {
            for (int j = 0; j < original_path.size(); j++) {
                if ((config->weight_selection_method == 0) ||
                    (config->weight_selection_method == 1 && original_path[j]->ignore) ||
                    (config->weight_selection_method == 2 && sample_path_set.find(original_path[j]) == sample_path_set.end())
                ) {
                   
                    if(num < 0)
                        original_path[j]->weight +=  abs(num)*10; 
                    else 
                    original_path[j]->weight += (sample_average / original_average - 1) * config->learning_rate;
                    original_path[j]->num_of_updates++;
                    num_of_edges_updated++;
                }
            }
            num_updates++;
        }

    
       
    }

    //Creating a histogram the accumlative change in the frequency in which the edges are being updated 
     if(!config->runs_prefix.empty()){
        int* count_updates = new int [20];
        std::fill(count_updates, count_updates + 20, 0);
        for(int j = 0; j < config->num_nodes ; j++){
                for(int k = 0; k < hnsw->mappings[j][0].size(); k++){
                Edge& edge = hnsw->mappings[j][0][k];
                if(edge.num_of_updates == 0 ) count_updates[0]++;
                else {
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

            // Used to check edge probability behaviour at each iteration
            // if (lowest_percentage > hnsw->mappings[i][0][k].probability_edge)
            //     lowest_percentage = hnsw->mappings[i][0][k].probability_edge;

            // if(max_probability < hnsw->mappings[i][0][k].probability_edge)
            //     max_probability = hnsw->mappings[i][0][k].probability_edge;
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

        // std::cout << std::setprecision(11);
        // cout << "left: " << left << " Right " << right << " MID" << mid << endl;
    }

    return left + (right - left) / 2;
}

// Load training set from training file or randomly generate them from nodes
void load_training(Config* config, float** nodes, float** training) {
    std::random_device rd;
    mt19937 gen(rd());
   
    if ( !config->generate_our_training && config->training_file != "") {
        if (config->query_file.size() >= 6 && config->training_file.substr(config->training_file.size() - 6) == ".fvecs") {
            if (config->generate_ratio > 0) {
                int num_loaded = config->num_training * (1.0 - config->generate_ratio);
                load_fvecs(config->training_file, "training", training, num_loaded, config->dimensions, config->groundtruth_file != "");
                normal_distribution<float> dis(config->gen_min, config->gen_max);
                for (int i = num_loaded; i < config->num_training; i++) {
                    training[i] = new float[config->dimensions];
                    for (int j = 0; j < config->dimensions; j++) {
                        training[i][j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
                    }
                }
                return;
            }
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

    if (!config->generate_our_training && config->load_file == "") {
        // Generate random training nodes
        cout << "Generating " << config->num_training << " random training points" << endl;
        normal_distribution<float> dis(config->gen_min, config->gen_max);

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
    normal_distribution<float>* dis_array = new normal_distribution<float>[config->dimensions];
    for (int i = 0; i < config->dimensions; i++) {
        dis_array[i] = normal_distribution<float>(lower_bound[i], upper_bound[i]);
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



void test_queue(){
    auto compare = [](Edge* lhs, Edge* rhs) { return lhs->distance > rhs->distance; };
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> candidates;
    priority_queue<Edge*, vector<Edge*>, decltype(compare)> candidates_edges(compare);
    for(int i =0; i< 50; i++ ){
        mt19937 gen(10);
        normal_distribution<float> dis(0, 1000);
        float distance = dis(gen);
        candidates.emplace(make_pair(distance, i));
        Edge* new_Edge = new Edge(i, distance);
        candidates_edges.emplace(new_Edge);

       if(candidates_edges.top()->target != candidates.top().second)
            cerr << "edge does not match node"  << candidates_edges.top()->target << ", " << candidates.top().second<< endl;
        else 
            cout << "it's working" <<endl;
    }
    exit(EXIT_FAILURE);

}