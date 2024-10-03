#include <iostream>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <float.h>
#include <set>
#include <limits>
#include "hnsw.h"
#include "pairingHeap.h"

using namespace std;

ofstream* debug_file = NULL;

int correct_nn_found = 0;
ofstream* when_neigh_found_file;


Edge::Edge() : target(-1), distance(-1), weight(50), ignore(false), probability_edge(0.5), num_of_updates(0), stinky(0), benefit(0), cost(0), prev_edge(nullptr){}

Edge::Edge(int target, float distance, int initial_cost, int initial_benefit) : target(target), distance(distance),
    weight(50), ignore(false), probability_edge(0.5), num_of_updates(0), stinky(0), benefit(initial_cost), cost(initial_benefit), prev_edge(nullptr){}

HNSW::HNSW(Config* config, float** nodes) : nodes(nodes), num_layers(1), num_nodes(config->num_nodes),
           num_dimensions(config->dimensions), entry_point(0), normal_factor(1 / -log(config->scaling_factor)),
           gen(config->insertion_seed), dis(0.0000001, 0.9999999), total_path_size(0), layer0_dist_comps_per_q(0), candidates_without_if(0),candidates_size(0) {
    reset_statistics();
    mappings.resize(num_nodes);
    mappings[0].resize(1);
}

//static 
std::map<int,std::vector<int>> HNSW::candidate_popping_times;  

float termination_alpha = 0;
float termination_alpha2 = 0 ;
float bw_break = 0;


void HNSW::reset_statistics() {
    layer0_dist_comps = 0;
    upper_dist_comps = 0;
    processed_neighbors = 0;
    total_neighbors = 0;
    num_distance_termination = 0;
    num_original_termination = 0;
    candidates_popped = 0;
    candidates_size = 0 ;
    candidates_without_if = 0;
    percent_neighbors.clear();
}

/**
 * Alg 1
 * INSERT(hnsw, q, M, Mmax, efConstruction, mL)
 * Note: max_con is not used for layer 0, instead max_connections_0 is used
*/
void HNSW::insert(Config* config, int query) {
    vector<pair<float, int>> entry_points;
    vector<Edge*> path;
    entry_points.reserve(config->ef_construction);
    int top = num_layers - 1;

    // Get node layer
    int node_layer = -log(dis(gen)) * normal_factor;
    mappings[query].resize(node_layer + 1);

    // Update layer count
    if (node_layer > top) {
        num_layers = node_layer + 1;
        if (config->debug_insert)
            cout << "Layer count increased to " << num_layers << endl;
    }

    float dist = calculate_distance(nodes[query], nodes[entry_point], num_dimensions, top);
    entry_points.push_back(make_pair(dist, entry_point));

    if (config->debug_insert)
        cout << "Inserting node " << query << " at layer " << node_layer << " with entry point " << entry_points[0].second << endl;

    // Find the closest point in each layer using search_layer
    for (int layer = top; layer >= node_layer + 1; layer--) {
        search_layer(config, nodes[query], path, entry_points, 1, layer);

        if (config->debug_insert)
            cout << "Closest point at layer " << layer << " is " << entry_points[0].second << " (" << entry_points[0].first << ")" << endl;
    }

    int max_connections = config->max_connections;
    for (int layer = min(top, node_layer); layer >= 0; layer--) {
        if (layer == 0)
            max_connections = config->max_connections_0;

        // Get nearest elements
        search_layer(config, nodes[query], path, entry_points, config->ef_construction, layer);
        // Choose opt_con number of neighbors out of candidates
        int num_neighbors = min(config->optimal_connections, static_cast<int>(entry_points.size()));
        // Initialize vector for node neighbors
        vector<Edge>& neighbors = mappings[query][layer];
        neighbors.reserve(max_connections + 1);
        neighbors.resize(num_neighbors);
        
        // Connect node to neighbors
        if (config->use_heuristic) {
            vector<Edge> candidates(entry_points.size());
            for (int i = 0; i < entry_points.size(); i++) {
                candidates[i] = Edge(entry_points[i].second, entry_points[i].first, config->initial_cost, config->initial_benefit);
            }
            select_neighbors_heuristic(config, nodes[query], candidates, num_neighbors, layer);
            for (int i = 0; i < num_neighbors; i++) {
                neighbors[i] = candidates[i];
            }
        } else {
            for (int i = 0; i < min(config->optimal_connections, (int)entry_points.size()); i++) {
                neighbors[i] = Edge(entry_points[i].second, entry_points[i].first, config->initial_cost, config->initial_benefit);
            }
        }

        // Print node neighbors
        if (config->debug_insert) {
            cout << "Neighbors at layer " << layer << " are ";
            for (auto n_pair : neighbors)
                cout << n_pair.target << " (" << n_pair.distance << ") ";
            cout << endl;
        }

        // Connect neighbors to node
        for (auto n_pair : neighbors) {
            vector<Edge>& neighbor_mapping = mappings[n_pair.target][layer];
            // Place query in the correct position in neighbor_mapping
            float new_dist = calculate_distance(nodes[query], nodes[n_pair.target], num_dimensions, layer);
            auto new_edge = Edge(query, new_dist, config->initial_cost, config->initial_benefit);
            auto pos = lower_bound(neighbor_mapping.begin(), neighbor_mapping.end(), new_edge,
                [](const Edge& lhs, const Edge& rhs) { return lhs.distance < rhs.distance || (lhs.distance == rhs.distance && lhs.target < rhs.target); });
            neighbor_mapping.insert(pos, new_edge);
        }

        // Trim neighbor connections if needed
        for (auto n_pair : neighbors) {
            vector<Edge>& neighbor_mapping = mappings[n_pair.target][layer];
            if (neighbor_mapping.size() > max_connections) {
                if (config->use_heuristic) {
                    select_neighbors_heuristic(config, nodes[query], neighbor_mapping, max_connections, layer);
                } else {
                    neighbor_mapping.pop_back();
                }
            }
        }

        // Resize entry_points to 1
        if (config->single_ep_construction)
            entry_points.resize(1);
    }

    if (node_layer > top) {
        entry_point = query;
    }
}

/**
 * Alg 2
 * Search_layer(hnsw, q, M, Mmax, efConstruction, mL)
 * Note: closest point are saved to entry_points 
 *       , and the path taken is saved into path which can be the direct path or beam_search bath dependent on variable config->use_direct_path
 *         
*/
void HNSW::search_layer(Config* config, float* query, vector<Edge*>& path, vector<pair<float, int>>& entry_points, int num_to_return, int layer_num, bool is_querying, bool is_training, bool is_ignoring, int* total_cost) {
    // Initialize search structures
    auto compare = [](Edge* lhs, Edge* rhs) { return lhs->distance > rhs->distance || (lhs->distance == rhs->distance && lhs->target > rhs->target); };
    unordered_set<int> visited;
    // The two candidates will be mapped such that if node x is at top of candidates queue, then edge pointing to x will be at the top of candidates_edges 
    // This way when we explore node x's neighbors and want to add parent edge to those newly explored edges, we use candidates_edges to access node x's edge and assign it as parent edge. 
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> candidates;
    // PairingHeap<pair<float, int>> candidates;

    priority_queue<Edge*, vector<Edge*>, decltype(compare)> candidates_edges(compare);
    vector<Edge*> entry_point_edges;
    priority_queue<pair<float, int>> found;
    priority_queue<pair<float, int>> top_k;
    bool using_top_k = is_querying && layer_num == 0 && (config->use_hybrid_termination || config->use_distance_termination);
    pair<float, int> top_1;

    //checking time popping of elements in candidates queue
    int candidate_insertion_times[1000000];
    int current_popping_time = 0; 

    // Initialize search_layer statistics
    vector<int> when_neigh_found(config->num_return, -1);
    int nn_found = 0;
    if (is_querying && layer_num == 0 && (config->use_distance_termination || config->use_calculation_termination || config->use_hybrid_termination)){
        num_to_return = 100000;
    }
    if (layer_num == 0 && config->print_neighbor_percent) {
        processed_neighbors = 0;
        total_neighbors = 0;
    }
    path.clear();


    // Add entry points to search structures 
    for (const auto& entry : entry_points) {
        visited.insert(entry.second);
        candidates.emplace(entry);
        found.emplace(entry);

        if(config->export_candidate_popping_times){
            candidate_insertion_times[entry.second] = current_popping_time;
     }

        // Create an new edge pointing at entry point
        if (layer_num == 0 && is_training && config->use_direct_path){
            Edge* new_Edge = new Edge(entry.second, entry.first, config->initial_cost, config->initial_benefit);
            candidates_edges.emplace(new_Edge);
            entry_point_edges.push_back(new_Edge);
            path.push_back(new_Edge);
        }
        if (using_top_k) {
            top_k.emplace(entry);
            top_1 = entry;
        }
        // Check if entry point is in groundtruth and update statistics accordingly
        if ((config->use_groundtruth_termination || config->export_oracle) && is_querying && layer_num == 0) {
            auto loc = find(cur_groundtruth.begin(), cur_groundtruth.end(), entry.second);
            if (loc != cur_groundtruth.end()) {
                int index = distance(cur_groundtruth.begin(), loc);
                if(index >= 0 && index < when_neigh_found.capacity())
                    when_neigh_found[index] = layer0_dist_comps_per_q;
                ++nn_found;
                ++correct_nn_found;
                // Break early if all actual nearest neighbors are found
                if (config->use_groundtruth_termination && nn_found == config->num_return)
                    // candidates = PairingHeap<pair<float, int>>();
                    candidates = priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>>();
                    break;
            }
        }
    }
    


    int candidates_popped_per_q = 0;
    int iteration = 0;
    float far_dist = found.top().first;
    while (!candidates.empty()) {
        if (debug_file != NULL) {
            // Export search data
            *debug_file << "Iteration " << iteration << endl;
            for (int index : visited)
                *debug_file << index << ",";
            *debug_file << endl;

            priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> temp_candidates(candidates);
            while (!temp_candidates.empty()) {
                *debug_file << temp_candidates.top().second << ",";
                temp_candidates.pop();
            }
            *debug_file << endl;

            priority_queue<pair<float, int>> temp_found(found);
            while (!temp_found.empty()) {
                *debug_file << temp_found.top().second << ",";
                temp_found.pop();
            }
            *debug_file << endl;
        }
        ++iteration;

        // Get the furthest distance element in found and closest element in candidates
        far_dist =  using_top_k ? far_dist: found.top().first;
        int closest = candidates.top().second;
        float close_dist = candidates.top().first;
        
        
        if(config->export_candidate_popping_times && is_querying && layer_num == 0){
            if(current_popping_time%config->cand_out_step == 0)
                candidate_popping_times[current_popping_time].push_back(candidate_insertion_times[closest]); //maybe replace with current time - insertion time 
            current_popping_time++;

        }


        candidates.pop();
        Edge* closest_edge;

        //for dirrect path
        if (layer_num == 0 && is_training && config->use_direct_path) {
            closest_edge = candidates_edges.top();
            candidates_edges.pop();
        }

        //for alternative beam width termination 
        if (layer_num == 0) {
            ++candidates_popped;
            ++candidates_popped_per_q;
        }

        // If terminating, log statistics and break
        if (should_terminate(config, top_k, top_1, close_dist, far_dist, is_querying, layer_num, candidates_popped_per_q)) {
            if (is_querying && layer_num == 0 && config->use_hybrid_termination){
                if (candidates_popped_per_q > config->ef_search)
                    num_original_termination++;
                else 
                    num_distance_termination++;
            }
            break;
        }

        // Explore neighbors of closest discovered element in the layer
        vector<Edge>& neighbors = mappings[closest][layer_num];
        for (auto& neighbor_edge : neighbors) {
            int neighbor = neighbor_edge.target;
            if (config->print_neighbor_percent && layer_num == 0) {
                ++total_neighbors;
            }
            candidates_without_if++;
            // Traverse newly discovered neighbor if we don't ignore it
            bool should_ignore = config->use_dynamic_sampling ? (dis(gen) < (1 -neighbor_edge.probability_edge)) : neighbor_edge.ignore;
            if (!(is_training && is_ignoring && should_ignore) && visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);

                //Debugging
                if (config->print_neighbor_percent && layer_num == 0) {
                    ++processed_neighbors;
                    if (total_neighbors == config->interval_for_neighbor_percent) {
                        percent_neighbors.push_back(static_cast<double>(processed_neighbors) / total_neighbors);
                        processed_neighbors = 0;
                        total_neighbors = 0;
                    }
                }


                // Add cost point to neighbor's edge if we are training
                if (is_training && config->use_stinky_points)
                    neighbor_edge.stinky -= config->stinky_value;
                if (is_training && config->use_cost_benefit) {
                    neighbor_edge.cost += 1;
                    if (total_cost != nullptr) {
                        *total_cost += 1;
                    }
                }
                
                // Add neighbor to structures if its distance to query is less than furthest found distance or beam structure isn't full
                float far_inner_dist = using_top_k ? far_dist : found.top().first;
                float neighbor_dist = calculate_distance(query, nodes[neighbor], num_dimensions, layer_num);
                if (neighbor_dist < far_inner_dist || found.size() < num_to_return) {
                    candidates.emplace(make_pair(neighbor_dist, neighbor));
                    
                    if(config->export_candidate_popping_times){
                        candidate_insertion_times[neighbor] = current_popping_time;
                    }


                    candidates_size++;
                    if (using_top_k) {
                        top_k.emplace(neighbor_dist, neighbor);
                        if(neighbor_dist > far_dist)
                            far_dist = neighbor_dist;
                        if (neighbor_dist < top_1.first) {
                            top_1 = make_pair(neighbor_dist, neighbor);
                        }
                        if (top_k.size() > config->num_return)
                            top_k.pop();
                    }
                    else{
                        found.emplace(neighbor_dist, neighbor);
                        if (found.size() > num_to_return){
                            found.pop();
                        }
                    }

                    if (layer_num == 0) {
                        path.push_back(&neighbor_edge);
                    }

                    //For Grasp training with Direct Path
                    if (layer_num == 0 && is_training && config->use_direct_path) {
                        neighbor_edge.distance = neighbor_dist;
                        neighbor_edge.prev_edge = closest_edge;
                        candidates_edges.emplace(&neighbor_edge);
                    }

                    // Check if entry point is in groundtruth and update statistics accordingly
                    //Used for oracle terminations (explicitly for oracle_1 and helps gather information for oracle_2)
                    if ((config->use_groundtruth_termination || config->export_oracle) && is_querying && layer_num == 0) {
                        auto loc = find(cur_groundtruth.begin(), cur_groundtruth.end(), neighbor);
                        if (loc != cur_groundtruth.end()) {
                            int index = distance(cur_groundtruth.begin(), loc);
                            when_neigh_found[index] = layer0_dist_comps_per_q;
                            ++nn_found;
                            ++correct_nn_found;
                            // Break early if all actual nearest neighbors are found
                            if (config->use_groundtruth_termination && nn_found == config->num_return)
                                // candidates = PairingHeap<pair<float, int>>();
                                candidates = priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>>();
                                break;
                        }
                    }

                    
                }
            }
        }
    }
   
    // Copy found into entry_points
   
    size_t idx = using_top_k ? top_k.size() :  config->single_ep_query || layer_num == 0 ? found.size() : min(config->k_upper, static_cast<int>(found.size()));
    entry_points.clear();
    entry_points.resize(idx);
    while (idx > 0) {
        --idx;

        if( (is_querying && layer_num == 0 && (config->use_hybrid_termination || config->use_distance_termination)) ){
            entry_points[idx] = make_pair(top_k.top().first, top_k.top().second);
            top_k.pop();
                
        }
        else{
            entry_points[idx] = make_pair(found.top().first, found.top().second);
            found.pop();
        }
    }
    // Calculate direct path
    if (config->use_direct_path && layer_num == 0 && is_training) {
        find_direct_path(path, entry_points);
        for (Edge* edge : entry_point_edges) {
            delete edge;
        }
    }
    // Export when_neigh_found data
    if (config->export_oracle && is_querying && layer_num == 0 && when_neigh_found_file != nullptr) {
        for (int i = 0; i < when_neigh_found.size(); ++i) {
            *when_neigh_found_file << when_neigh_found[i] << " ";
        }
    }
}

/**
 * Alg 4
 * SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections)
 * Given a query and candidates, set candidates to the num_to_return best candidates according to a heuristic
 */
void HNSW::select_neighbors_heuristic(Config* config, float* query, vector<Edge>& candidates, int num_to_return, int layer_num, bool extend_candidates, bool keep_pruned) {
    // Initialize output vector, consider queue, and discard queue
    auto compare = [](const Edge& lhs, const Edge& rhs) { return lhs.distance > rhs.distance; };
    vector<Edge> output;
    priority_queue<Edge, vector<Edge>, decltype(compare)> considered(compare);
    priority_queue<Edge, vector<Edge>, decltype(compare)> discarded(compare);
    for (const Edge& candidate : candidates) {
        considered.emplace(candidate);
    }

    // Extend candidate list by their neighbors
    if (extend_candidates) {
        for (auto candidate : candidates) {
            for (const Edge& neighbor : mappings[candidate.target][layer_num]) {
                // Make sure neighbor isn't already being considered
                bool is_found = false;
                priority_queue<Edge, vector<Edge>, decltype(compare)> temp_considered(considered);
                while (!temp_considered.empty()) {
                    if (temp_considered.top().target == neighbor.target) {
                        is_found = true;
                        break;
                    }
                    temp_considered.pop();
                }
                if (!is_found) {
                    considered.emplace(neighbor);
                }
            }
        }
    }

    // Add considered element to output if it is closer to query than to other output elements
    while (!considered.empty() && output.size() < num_to_return) {
        const Edge& closest = considered.top();
        float query_distance = calculate_distance(nodes[closest.target], query, num_dimensions, layer_num);
        bool is_closer_to_query = true;
        for (auto n_pair : output) {
            if (query_distance >= calculate_distance(nodes[closest.target], nodes[n_pair.target], num_dimensions, layer_num)) {
                is_closer_to_query = false;
                break;
            }
        }
        if (is_closer_to_query) {
            output.emplace_back(closest);
        } else {
            discarded.emplace(closest);
        }
        considered.pop();
    }

    // Add discarded elements until output is large enough
    if (keep_pruned) {
        while (!discarded.empty() && output.size() < num_to_return) {
            output.emplace_back(discarded.top());
            discarded.pop();
        }
    }

    // Set candidates to output
    candidates.clear();
    candidates.resize(output.size());
    for (int i = 0; i < output.size(); i++) {
        candidates[i] = output[i];
    }
}

/**
 * Alg 5
 * K-NN-SEARCH(hnsw, q, K, ef)
 * This also stores the traversed bottom-layer edges in the path vector
*/
vector<pair<float, int>> HNSW::nn_search(Config* config, vector<Edge*>& path, pair<int, float*>& query, int num_to_return, bool is_querying, bool is_training, bool is_ignoring, int* total_cost) {
    // Begin search at the top layer entry point
    vector<pair<float, int>> entry_points;
    entry_points.reserve(config->ef_search);
    int top = num_layers - 1;
    float dist = calculate_distance(query.second, nodes[entry_point], num_dimensions, top);
    entry_points.push_back(make_pair(dist, entry_point));
    if (config->debug_search)
        cout << "Searching for " << num_to_return << " nearest neighbors of node " << query.first << endl;

    // Find the closest point to the query at each upper layer
    for (int layer = top; layer >= 1; layer--) {
         if ((config->single_ep_query && !is_training) || (config->single_ep_training && is_training)) {
            search_layer(config, query.second, path, entry_points, 1, layer, is_querying);
        } else {
            search_layer(config, query.second, path, entry_points, config->ef_search_upper, layer, is_querying);
        }
        if (config->debug_search)
            cout << "Closest point at layer " << layer << " is " << entry_points[0].second << " (" << entry_points[0].first << ")" << endl;
    }

    // Search the bottom layer and conditionally enable loggers
    if (config->debug_query_search_index == query.first) {
        debug_file = new ofstream(config->runs_prefix + "query_search.txt");
    }
    search_layer(config, query.second, path, entry_points, config->ef_search, 0, is_querying, is_training, is_ignoring, total_cost);
    if (config->print_path_size) {
        total_path_size += path.size();
    }
    if (config->debug_query_search_index == query.first) {
        debug_file->close();
        delete debug_file;
        debug_file = NULL;
        cout << "Exported query search data to " << config->runs_prefix << "query_search.txt for query " << query.first << endl;
    }
    if (config->debug_search) {
        cout << "All closest points at layer 0 are ";
        for (auto n_pair : entry_points)
            cout << n_pair.second << " (" << n_pair.first << ") ";
        cout << endl;
    }

    // Return the closest num_return elements from entry_points
    entry_points.resize(min(entry_points.size(), (size_t)num_to_return));
    return entry_points;
}

/*
 * Finds the direct path to each nearest neighbor stored in entry_points by
 * backtracking along the beam searched path until a nullptr is reached. This
 * sets the path to the newly found path and deallocates the entry points.
 */
void HNSW::find_direct_path(vector<Edge*>& path, vector<pair<float, int>>& entry_points) {
    set<Edge*> direct_path;
    for (int i = 0; i < entry_points.size(); i++) {
        // Find the edge point to approximate nearest neighbor
        Edge* current = nullptr;
        for (auto edge : path) {
            if (edge->target == entry_points[i].second) {
                current = edge;
                break;
            }
        }

        if (current == nullptr) {
            cerr << "Start edge wasn't found, can't find strict path for entry point " << i << endl;
            break;
        } else {
            // Traverse back through the path
            int size = 0;
            while (size < path.size() && current->prev_edge != nullptr && direct_path.find(current) == direct_path.end()) {
                direct_path.insert(current);
                current = current->prev_edge;
                ++size;
            }
        }
    }
    // Copy direct_path into path parameter
    vector<Edge*> direct_path_vector(direct_path.begin(), direct_path.end());
    path = direct_path_vector;
}

// Returns whether or not to terminate from search_layer
bool HNSW::should_terminate(Config* config, priority_queue<pair<float, int>>& top_k, pair<float, int>& top_1, float close_squared, float far_squared,  bool is_querying, int layer_num,int candidates_popped_per_q) {
    // Evaluate beam-width-based criteria
    bool beam_width_original = close_squared > far_squared;
    // Use candidates_popped as a proxy for beam-width
    bool beam_width_1 = candidates_popped_per_q > config->ef_search;
    bool beam_width_2 =  false;
    bool alpha_distance_1 = false;
    bool alpha_distance_2 = false;
    bool num_of_dist_1 = false;

    // Evaluate distance-based criteria
    if (is_querying && layer_num == 0 && (config->use_hybrid_termination || config->use_distance_termination)) {
        float close = sqrt(close_squared);
        float threshold;
        threshold = 2 * sqrt(top_k.top().first) + sqrt(top_1.first);
        
        
        alpha_distance_1 = top_k.size() >= config->num_return && close > termination_alpha * threshold;
        
      
        // Evaluate break points
        if (config->use_latest && config->use_break) {
            alpha_distance_2 = top_k.size() >= config->num_return && close > termination_alpha2 * threshold;
            beam_width_2 = candidates_popped_per_q > bw_break;

        }

    }
    // Return whether to terminate using config flags
    if (!is_querying || layer_num > 0) {
        return beam_width_original;
    } else if (config->use_hybrid_termination && config->use_latest) {
        return (alpha_distance_1 && beam_width_1) || alpha_distance_2 ||  beam_width_2;
    } else if (config->use_hybrid_termination) {
        return alpha_distance_1 || beam_width_1;
    } else if (config->use_distance_termination) {
        return alpha_distance_1;
    } else if(config->use_calculation_termination) {
        return  config->calculations_per_query < layer0_dist_comps_per_q;
    } else {
        return beam_width_original;
    }
}


 void HNSW::calculate_termination(Config *config){
        float estimated_distance_calcs = config->bw_slope != 0 ? (config->ef_search - config->bw_intercept) / config->bw_slope : 1;
        termination_alpha = config->use_distance_termination ? config->termination_alpha : config->alpha_coefficient * log(estimated_distance_calcs) + config->alpha_intercept;

        estimated_distance_calcs *=config->alpha_break;
        termination_alpha2 = config->alpha_coefficient * log(estimated_distance_calcs) + config->alpha_intercept;


         ifstream histogram = ifstream(config->metric_prefix + "_median_percentiles.txt");

            if(!histogram.fail() && config->use_median_break){
                string info;
                int line  = find(config->benchmark_ef_search.begin(),config->benchmark_ef_search.end(), config->ef_search) - config->benchmark_ef_search.begin();
                int index = find(config->benchmark_median_percentiles.begin(),config->benchmark_median_percentiles.end(), config->breakMedian) - config->benchmark_median_percentiles.begin()+1; 
                int distance_termination = 0;
                while(line != 0 && getline(histogram,info)){
                    line--;
                }

                while(histogram >> info && index!= 0) {
                    index--;
            
                }
                estimated_distance_calcs = stoi(info);
            } 

        bw_break = static_cast<int>(config->bw_slope  * estimated_distance_calcs + config->bw_intercept);
        // cout << "bw break is: " << bw_break << ", for estimated calc = " << estimated_distance_calcs; 
    }


// Searches for each query using the HNSW graph
void HNSW::search_queries(Config* config, float** queries) {
    // Initialize log files
    ofstream* export_file = NULL;
    if (config->export_queries)
        export_file = new ofstream(config->runs_prefix + "queries.txt");
    ofstream* indiv_file = NULL;
    if (config->export_indiv)
        indiv_file = new ofstream(config->runs_prefix + "indiv.txt");
    if (config->export_oracle)
        when_neigh_found_file = new ofstream(config->oracle_file);

    // Load actual nearest neighbors
    bool use_groundtruth = config->groundtruth_file != "";
    if (use_groundtruth && config->query_file == "") {
        cout << "Warning: Groundtruth file will not be used because queries were generated" << endl;
        use_groundtruth = false;
    }
    vector<vector<int>> actual_neighbors;
    if (use_groundtruth) {
        load_ivecs(config->groundtruth_file, actual_neighbors, config->num_queries, config->num_return);
    } else {
        knn_search(config, actual_neighbors, nodes, queries);
    }

    // Initialize calculations per query and oracle calculations
    vector<int> counts_calcs;
    for (int i = 0; i < 20; i++) {
        counts_calcs.push_back(0);
    }
    vector<pair<int, int>> nn_calculations;
    if (config->use_calculation_oracle) {
        load_oracle(config, nn_calculations);
    }
    int oracle_distance_calcs = 0;

    int total_found = 0;
    reset_statistics();
    for (int i = 0; i < config->num_queries; ++i) {
        // Obtain the query and optionally check if it's too expensive according to the oracle
        float* query = config->use_calculation_oracle ? queries[nn_calculations[i].second] : queries[i];
        if (config->use_calculation_oracle) {
            oracle_distance_calcs += nn_calculations[i].first;
        }
        if (oracle_distance_calcs > config->oracle_termination_total) {
            break;
        }
        pair<int, float*> query_pair = make_pair(i, query);

        cur_groundtruth = actual_neighbors[i];
        layer0_dist_comps_per_q = 0;
        vector<Edge*> path;
        vector<pair<float, int>> found = nn_search(config, path, query_pair, config->num_return);

        // Update log files
        if (config->export_calcs_per_query) {
            ++counts_calcs[std::min(19, layer0_dist_comps_per_q / config->interval_for_calcs_histogram)];
        }
        if (config->export_oracle)
            *when_neigh_found_file << endl;
        if (config->print_results) {
            // Print out found
            cout << "Found " << found.size() << " nearest neighbors of [" << query_pair.second[0];
            for (int dim = 1; dim < num_dimensions; ++dim)
                cout << " " << query_pair.second[dim];
            cout << "] : ";
            for (auto n_pair : found)
                cout << n_pair.second << " ";
            cout << endl;
            // Print path
            cout << "Path taken: ";
            for (Edge* edge : path) {
                cout << edge->target << " ";
            }
            cout << endl;
        }
        if (config->print_actual) {
            // Print out actual
            cout << "Actual " << config->num_return << " nearest neighbors of [" << query_pair.second[0];
            for (int dim = 1; dim < num_dimensions; ++dim)
                cout << " " << query_pair.second[dim];
            cout << "] : ";
            for (int index : actual_neighbors[i])
                cout << index << " ";
            cout << endl;
        }

        // Print accuracy thus far
        if (config->print_indiv_found || config->print_total_found || config->export_indiv) {
            unordered_set<int> actual_set(actual_neighbors[i].begin(), actual_neighbors[i].end());
            int matching = 0;
            for (auto n_pair : found) {
                if (actual_set.find(n_pair.second) != actual_set.end())
                    ++matching;
            }
            if (config->print_indiv_found)
                cout << "Found " << matching << " (" << matching /  (double)config->num_return * 100 << "%) for query " << i << endl;
            if (config->print_total_found)
                total_found += matching;
            if (config->export_indiv)
                *indiv_file << matching / (double)config->num_return << " " << layer0_dist_comps << " " << upper_dist_comps << endl;
        }

        // Export the query used
        if (config->export_queries) {
            *export_file << "Query " << i+1 << endl << query_pair.second[0] << endl;
            if (config->num_return == 1) {
                for (int j =0; j< found.size(); j++) {
                    *export_file << found[j].second << "," << found[j].first << endl;
                    *export_file << cur_groundtruth[j];
                    if(found[j].second != cur_groundtruth[j]){ 
                        *export_file << "," << calculate_l2_sq(queries[i], nodes[actual_neighbors[i][j]], config->dimensions);
                    }
                    *export_file<< endl;
                }
            } else {
                for (int dim = 1; dim < num_dimensions; ++dim)
                    *export_file << "," << query_pair.second[dim];
                *export_file << endl;
                for (auto n_pair : found)
                    *export_file << n_pair.second << ",";
                *export_file << endl;
                for (int index : cur_groundtruth)
                    *export_file << index << " ";
                *export_file << endl;
                
                for (Edge* edge : path) {
                    *export_file << edge->target << ",";
                }
            }
            *export_file << endl;
        }
    }

    // Finalize log files
    if (config->export_calcs_per_query) {
        ofstream histogram = ofstream(config->runs_prefix + "histogram_calcs_per_query.txt", std::ios::app);
        for (int i = 0; i < 20; ++i) {
            histogram << counts_calcs[i] << ",";
        }
        histogram << endl;
        histogram.close();
    }
    if (config->export_oracle) {
        cout << "Total neighbors found (groundtruth comparison): " << correct_nn_found << " (" << correct_nn_found / (double)(config->num_queries * config->num_return) * 100 << "%)" << endl;
    }
    if (config->print_total_found) {
        cout << "Total neighbors found: " << total_found << " (" << total_found / (double)(config->num_queries * config->num_return) * 100 << "%)" << endl;
    }
    cout << "Finished search" << endl;
    if (export_file != NULL) {
        export_file->close();
        delete export_file;
        cout << "Exported queries to " << config->runs_prefix << "queries.txt" << endl;
    }
    if (indiv_file != NULL) {
        indiv_file->close();
        delete indiv_file;
        cout << "Exported individual query results to " << config->runs_prefix << "indiv.txt" << endl;
    }
    if (config->export_oracle) {
        when_neigh_found_file->close();
        delete when_neigh_found_file;
        cout << "Exported when neighbors were found to " << config->oracle_file << endl;
    }

  
}

// Gets all edges in a specific layer
vector<Edge*> HNSW::get_layer_edges(Config* config, int layer) {
    vector<Edge*> edges;
    for (int i = 0; i < config->num_nodes; i++) {
        // If node in adjacency list has at least 'layer' layers, add its edges to the output
        if (mappings[i].size() - 1 >= layer) {
            for (int j = 0; j < mappings[i][layer].size(); j++) {
                edges.push_back(&mappings[i][layer][j]);
            }
        }
    }
    return edges;
}

// Computes the average ratio of closed triplets to total triplets
float HNSW::calculate_global_clustering_coefficient() {
    int num_closed_triplets=0;
    int num_triplets=0;
    for (int i = 0; i < mappings.size(); ++i) {
        vector<Edge>& first_neighbors = mappings[i][0];
        // Convert vector of edges to set of nodes
        unordered_set<int> target_set;
        for (int j = 0; j < first_neighbors.size(); ++j) {
            target_set.insert(first_neighbors[j].target);
        }
        // Count the number of neighbors' neighbors that are adjacent to node i
        for (int j = 0; j < first_neighbors.size(); ++j) {
            vector<Edge>& second_neighbors = mappings[first_neighbors[j].target][0];
            for (int k = 0; k < second_neighbors.size(); ++k) {
                if (target_set.find(second_neighbors[k].target) != target_set.end()) {
                    ++num_closed_triplets;
                } else {
                    vector<Edge>& third_neighbors = mappings[second_neighbors[k].target][0];
                    for (int l = 0; l < third_neighbors.size(); ++l) {
                        if (third_neighbors[l].target == i) {
                            ++num_closed_triplets;
                        }
                    }
                }
                ++num_triplets;
            }
        }
    }
    return static_cast<float>(num_closed_triplets) / num_triplets;
}

// Computes the average ratio of actual connected neighbors to possible connected neighbors
float HNSW::calculate_average_clustering_coefficient() {
    float coefficient = 0;
    for (int i = 0; i < mappings.size(); ++i) {
        vector<Edge>& first_neighbors = mappings[i][0];
        // Convert vector of edges to set of nodes
        unordered_set<int> target_set;
        for (int j = 0; j < first_neighbors.size(); ++j) {
            target_set.insert(first_neighbors[j].target);
        }
        // Count the number of neighbors' neighbors that are adjacent to node i
        int num_connected = 0;
        for (int j = 0; j < first_neighbors.size(); ++j) {
            vector<Edge>& second_neighbors = mappings[first_neighbors[j].target][0];
            for (int k = 0; k < second_neighbors.size(); ++k) {
                if (target_set.find(second_neighbors[k].target) != target_set.end()) {
                    ++num_connected;
                }
            }
        }
        // Add the current coefficient to the total if there is at least 2 neighbors
        if (first_neighbors.size() > 1) {
            coefficient += static_cast<float>(num_connected) / (first_neighbors.size() * (first_neighbors.size() - 1));
        }
    }
    return coefficient / mappings.size();
}

// Computes the distance between a and b and update dist_comps accordingly
float HNSW::calculate_distance(float* a, float* b, int size, int layer) {
    if (layer == 0){
        ++layer0_dist_comps;
        ++layer0_dist_comps_per_q;
    }
    else if (layer > 0)
        ++upper_dist_comps;
    return calculate_l2_sq(a, b, size);
}

std::ostream& operator<<(std::ostream& os, const HNSW& hnsw) {
    vector<int> nodes_per_layer(hnsw.num_layers);
    for (int i = 0; i < hnsw.num_nodes; ++i) {
        for (int j = 0; j < hnsw.mappings[i].size(); ++j)
            ++nodes_per_layer[j];
    }

    os << "Nodes per layer: " << endl;
    for (int i = 0; i < hnsw.num_layers; ++i)
        os << "Layer " << i << ": " << nodes_per_layer[i] << endl;
    os << endl;

    for (int i = 0; i < hnsw.num_layers; ++i) {
        os << "Layer " << i << " connections: " << endl;
        for (int j = 0; j < hnsw.num_nodes; ++j) {
            if (hnsw.mappings[j].size() <= i)
                continue;

            os << j << ": ";
            for (auto n_pair : hnsw.mappings[j][i])
                os << n_pair.target << " ";
            os << endl;
        }
    }
    return os;
}

void HNSW::from_files(Config* config, bool is_benchmarking) {
    // Open files
    ifstream graph_file(config->loaded_graph_file);
    ifstream info_file(config->loaded_info_file);
    cout << "Loading saved graph from " << config->loaded_graph_file << endl;
    if (!graph_file) {
        cout << "File " << config->loaded_graph_file << " not found!" << endl;
        return;
    }
    if (!info_file) {
        cout << "File " << config->loaded_info_file << " not found!" << endl;
        return;
    }

    // Process info file
    int opt_con, max_con, max_con_0, ef_con;
    int num_nodes;
    int read_num_layers;
    long long construct_layer0_dist_comps;
    long long construct_upper_dist_comps;
    double construct_duration;
    info_file >> opt_con >> max_con >> max_con_0 >> ef_con;
    info_file >> num_nodes;
    info_file >> read_num_layers;
    if (is_benchmarking) {
        info_file >> construct_layer0_dist_comps;
        info_file >> construct_upper_dist_comps;
        info_file >> construct_duration;
    }
    num_layers = read_num_layers;

    // Verify config parameters
    if (num_nodes != config->num_nodes) {
        cout << "Mismatch between loaded and expected number of nodes" << endl;
        return;
    }
    if (opt_con != config->optimal_connections || max_con != config->max_connections ||
        max_con_0 != config->max_connections_0 || ef_con != config->ef_construction) {
        cout << "Mismatch between loaded and expected construction parameters" << endl;
        return;
    }

    // Process graph file
    auto start = chrono::high_resolution_clock::now();
    cout << "Loading graph with construction parameters: "
         << config->optimal_connections << ", " << config->max_connections << ", "
         << config->max_connections_0 << ", " << config->ef_construction << endl;
    for (int i = 0; i < num_nodes; ++i) {
        int layers;
        graph_file.read(reinterpret_cast<char*>(&layers), sizeof(layers));
        mappings[i].resize(layers);
        // Load each layer
        for (int j = 0; j < layers; ++j) {
            int num_neighbors;
            graph_file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
            mappings[i][j].reserve(num_neighbors);
            // Load each neighbor
            for (int k = 0; k < num_neighbors; ++k) {
                int index;
                float distance;
                graph_file.read(reinterpret_cast<char*>(&index), sizeof(index));
                graph_file.read(reinterpret_cast<char*>(&distance), sizeof(distance));
                mappings[i][j].emplace_back(Edge(index, distance, config->initial_cost, config->initial_benefit));
            }
        }
    }
    int read_entry_point;
    graph_file.read(reinterpret_cast<char*>(&read_entry_point), sizeof(read_entry_point));
    entry_point = read_entry_point;

    // Conditionally print benchmark data
    if (is_benchmarking) {
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Load time: " << duration / 1000.0 << " seconds, ";
        cout << "Construction time: " << construct_duration << " seconds, ";
        cout << "Distance computations (layer 0): " << construct_layer0_dist_comps <<", ";
        cout << "Distance computations (top layers): " << construct_upper_dist_comps << endl;
    }
}

void HNSW::to_files(Config* config, const string& graph_name, long int construction_duration) {
    // Export graph to file
    ofstream graph_file(config->runs_prefix + "graph_" + graph_name + ".bin");

    // Export edges
    for (int i = 0; i < num_nodes; ++i) {
        // Write number of layers
        int layers = mappings[i].size();
        graph_file.write(reinterpret_cast<const char*>(&layers), sizeof(layers));

        // Write each layer
        for (int j = 0; j < layers; ++j) {
            // Write number of neighbors
            int num_neighbors = mappings[i][j].size();
            graph_file.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));

            // Write index and distance of each neighbor
            for (int k = 0; k < num_neighbors; ++k) {
                auto n_pair = mappings[i][j][k];
                graph_file.write(reinterpret_cast<const char*>(&n_pair.target), sizeof(n_pair.target));
                graph_file.write(reinterpret_cast<const char*>(&n_pair.distance), sizeof(n_pair.distance));
            }
        }
    }
    // Save entry point
    graph_file.write(reinterpret_cast<const char*>(&entry_point), sizeof(entry_point));
    graph_file.close();

    // Export construction parameters
    ofstream info_file(config->runs_prefix + "info_" + graph_name + ".txt");
    info_file << config->optimal_connections << " " << config->max_connections << " "
              << config->max_connections_0 << " " << config->ef_construction << endl;
    info_file << num_nodes << endl;
    info_file << num_layers << endl;
    info_file << layer0_dist_comps << endl;
    info_file << upper_dist_comps << endl;
    info_file << construction_duration << endl;

    cout << "Exported graph to " << config->runs_prefix + "graph_" + graph_name + ".bin" << endl;
}
