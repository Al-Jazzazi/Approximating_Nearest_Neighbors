#include <iostream>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <immintrin.h>
#include <unordered_set>
#include "hnsw.h"
#include <float.h>
#include <set>

using namespace std;

ofstream* debug_file = NULL;

int correct_nn_found = 0;
bool log_neighbors = false;
vector<int> cur_groundtruth;
ofstream* when_neigh_found_file;

Edge::Edge() : target(-1), distance(-1), weight(50), ignore(false), probability_edge(0.5), num_of_updates(0), stinky(0), benefit(0), cost(0), prev_edge(nullptr){}

Edge::Edge(int target, float distance, int initial_cost, int initial_benefit) : target(target), distance(distance),
    weight(50), ignore(false), probability_edge(0.5), num_of_updates(0), stinky(0), benefit(initial_cost), cost(initial_benefit), prev_edge(nullptr){}

HNSW::HNSW(Config* config, float** nodes) : nodes(nodes), num_layers(0), num_nodes(config->num_nodes),
           num_dimensions(config->dimensions), normal_factor(1 / -log(config->scaling_factor)),
           gen(config->insertion_seed), dis(0.0000001, 0.9999999), total_path_size(0) {
    reset_statistics();
}

void HNSW::reset_statistics() {
    layer0_dist_comps = 0;
    upper_dist_comps = 0;
    actual_beam_width = 0;
    processed_neighbors = 0;
    total_neighbors = 0;
    num_distance_termination = 0;
    num_original_termination = 0;
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

    float dist = calculate_l2_sq(this, top, nodes[query], nodes[entry_point], num_dimensions);
    entry_points.push_back(make_pair(dist, entry_point));

    if (config->debug_insert)
        cout << "Inserting node " << query << " at layer " << node_layer << " with entry point " << entry_points[0].second << endl;

    // Get closest element by using search_layer to find the closest point at each layer
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
        // Initialize mapping vector
        vector<Edge>& neighbors = mappings[query][layer];
        neighbors.reserve(max_connections + 1);
        neighbors.resize(num_neighbors);
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

        if (config->debug_insert) {
            cout << "Neighbors at layer " << layer << " are ";
            for (auto n_pair : neighbors)
                cout << n_pair.target << " (" << n_pair.distance << ") ";
            cout << endl;
        }

        // Connect neighbors to this node
        for (auto n_pair : neighbors) {
            vector<Edge>& neighbor_mapping = mappings[n_pair.target][layer];

            // Place query in correct position in neighbor_mapping
            float new_dist = calculate_l2_sq(this, layer, nodes[query], nodes[n_pair.target], num_dimensions);
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

        if (config->single_ep_construction)
            // Resize entry_points to 1
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
void HNSW::search_layer(Config* config, float* query, vector<Edge*>& path, vector<pair<float, int>>& entry_points, int num_to_return, int layer_num, bool is_querying, bool is_training, bool is_ignoring) {
    // Initialize search structures
    auto compare = [](Edge* lhs, Edge* rhs) { return lhs->distance > rhs->distance || (lhs->distance == rhs->distance && lhs->target > rhs->target); };
    unordered_set<int> visited;
    // The two candidates will be mapped such that if node x is at top of candidates queue, then edge pointing to x will be at the top of candidates_edges 
    // This way when we explore node x's neighbors and want to add parent edge to those newly exlpored edges, we use candidates_edges to access node x's edge and assign it as parent edge. 
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> candidates;
    priority_queue<Edge*, vector<Edge*>, decltype(compare)> candidates_edges(compare);
    vector<Edge*> entry_point_edges;
    priority_queue<pair<float, int>> found;
    priority_queue<pair<float, int>> found_extension;
    priority_queue<pair<float, int>> top_k;
    pair<float, int> top_1;
    if (is_querying && layer_num == 0 && config->use_distance_termination && !config->combined_termination) {
        num_to_return = 100000;
    }

    // Initialize statistics
    vector<int> when_neigh_found(config->num_return, -1);
    int nn_found = 0;
    path.clear();
    if (layer_num == 0 && config->print_neighbor_percent) {
        processed_neighbors = 0;
        total_neighbors = 0;
    }
  
    //to not have found_extension empty 
    found_extension.emplace(-FLT_MAX, -1);
    // Add entry points to search structures
    for (const auto& entry : entry_points) {
        visited.insert(entry.second);
        candidates.emplace(entry);
        found.emplace(entry);
        // Create an new edge pointing at entry point
        if (layer_num == 0 && is_training && config->use_direct_path){
            Edge* new_Edge = new Edge(entry.second, entry.first, config->initial_cost, config->initial_benefit);
            candidates_edges.emplace(new_Edge);
            entry_point_edges.push_back(new_Edge);
            path.push_back(new_Edge);
        }
        if (is_querying && layer_num == 0 && (config->combined_termination || config->use_distance_termination)) {
            top_k.emplace(entry);
            top_1 = entry;
        }
        if (log_neighbors) {
            auto loc = find(cur_groundtruth.begin(), cur_groundtruth.end(), entry.second);
            if (loc != cur_groundtruth.end()) {
                // Get neighbor index (xth closest) and log distance comp
                int index = distance(cur_groundtruth.begin(), loc);
                when_neigh_found[index] = layer0_dist_comps;
                ++nn_found;
                ++correct_nn_found;
                if (config->gt_smart_termination && nn_found == config->num_return)
                    // End search
                    candidates = priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>>();
            }
        }
    }
    

   
    int iteration = 0;
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

        // Get the furthest element in found and closest element in candidates
        int furthest = found.top().second;
        float far_dist = found.top().first;
        int closest = candidates.top().second;
        float close_dist = candidates.top().first;
        candidates.pop();
        Edge* closest_edge;
        if(layer_num == 0 && is_training && config->use_direct_path){
            closest_edge = candidates_edges.top();
            candidates_edges.pop();
        }

        // If terminating, log statistics and break
        if (should_terminate(config, top_k, top_1, close_dist, far_dist, found_extension.top().first ,is_querying, layer_num)) {
            if (layer_num == 0) {
                actual_beam_width += found.size();
            }
            if(is_querying && layer_num == 0 && config->combined_termination){
                if(close_dist > far_dist)
                    num_original_termination++;
                else 
                    num_distance_termination++;
            }
            break;
        }

        // Explore neighbors of closest discovered element in HNSW layer
        vector<Edge>& neighbors = mappings[closest][layer_num];
        for (auto& neighbor_edge : neighbors) {
            int neighbor = neighbor_edge.target;
            if (config->print_neighbor_percent && layer_num == 0) {
                ++total_neighbors;
            }

            // Traverse newly discovered neighbor if we don't ignore it
            bool should_ignore = config->use_dynamic_sampling ? (dis(gen) < (1 -neighbor_edge.probability_edge)) : neighbor_edge.ignore;
            if (!(is_training && is_ignoring && should_ignore) && visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
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
                if (is_training && config->use_cost_benefit)
                    ++neighbor_edge.cost;
                
                // Add neighbor to structures if its distance to query is less than furthest found distance or beam structure isn't full
                float far_inner_dist = found.top().first;
                float neighbor_dist = calculate_l2_sq(this, layer_num, query, nodes[neighbor], num_dimensions);
                if (neighbor_dist < far_inner_dist || found.size() < num_to_return) {
                    candidates.emplace(neighbor_dist, neighbor);
                    found.emplace(neighbor_dist, neighbor);
                    if (is_querying && layer_num == 0 && (config->combined_termination || config->use_distance_termination)) {
                        top_k.emplace(neighbor_dist, neighbor);
                        if (neighbor_dist < top_1.first) {
                            top_1 = make_pair(neighbor_dist, neighbor);
                        }
                        if (top_k.size() > config->num_return)
                            top_k.pop();
                    }
                    if (layer_num == 0) {
                        path.push_back(&neighbor_edge);
                    }
                    if (layer_num == 0 && is_training && config->use_direct_path) {
                        neighbor_edge.distance = neighbor_dist;
                        neighbor_edge.prev_edge = closest_edge;
                        candidates_edges.emplace(&neighbor_edge);
                    }
                    if (log_neighbors) {
                        auto loc = find(cur_groundtruth.begin(), cur_groundtruth.end(), neighbor);
                        if (loc != cur_groundtruth.end()) {
                            // Get neighbor index (xth closest) and log distance comp
                            int index = distance(cur_groundtruth.begin(), loc);
                            when_neigh_found[index] = layer0_dist_comps;
                            ++nn_found;
                            ++correct_nn_found;
                            if (config->gt_smart_termination && nn_found == config->num_return)
                                // End search
                                candidates = priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>>();
                        }
                    }

                    // If priority queues are too large, remove the furthest
                    if (found.size() > num_to_return){
                        found.pop();
                        found_extension.emplace(far_dist, furthest);
                        if(found_extension.size() > num_to_return * config->efs_search_2)
                            found_extension.pop();
                    }
                } else if(config->use_latest && config->use_break && (neighbor_dist < found_extension.top().first || found_extension.size() <   num_to_return * config->efs_search_2)){
                    found_extension.emplace(neighbor_dist, neighbor);
                    if(found_extension.size() > num_to_return * config->efs_search_2)
                        found_extension.pop();
                }
    
            } 
           
        }
    }

    // Place found elements into entry_points
    size_t idx = layer_num == 0 || config->single_ep_query ? found.size() : min(config->k_upper, static_cast<int>(found.size()));
    entry_points.clear();
    entry_points.resize(idx);
    while (idx > 0) {
        --idx;
        entry_points[idx] = make_pair(found.top().first, found.top().second);
        found.pop();
    }
    // Calculate direct path
    if (config->use_direct_path && layer_num == 0 && is_training) {
        find_direct_path(path, entry_points);
        for (Edge* edge : entry_point_edges) {
            delete edge;
        }
    }
    // Export when_neigh_found data
    if (log_neighbors) {
        for (int i = 0; i < config->num_return; ++i) {
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
        float query_distance = calculate_l2_sq(this, layer_num, nodes[closest.target], query, num_dimensions);
        bool is_closer_to_query = true;
        for (auto n_pair : output) {
            if (query_distance >= calculate_l2_sq(this, layer_num, nodes[closest.target], nodes[n_pair.target], num_dimensions)) {
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
vector<pair<float, int>> HNSW::nn_search(Config* config, vector<Edge*>& path, pair<int, float*>& query, int num_to_return, bool is_querying, bool is_training, bool is_ignoring) {
    vector<pair<float, int>> entry_points;
    entry_points.reserve(config->ef_search);
    int top = num_layers - 1;
    float dist = calculate_l2_sq(this, top, query.second, nodes[entry_point], num_dimensions);
    entry_points.push_back(make_pair(dist, entry_point));

    if (config->debug_search)
        cout << "Searching for " << num_to_return << " nearest neighbors of node " << query.first << endl;

    // Get closest element by using search_layer to find the closest point at each layer
    for (int layer = top; layer >= 1; layer--) {
         if ((config->single_ep_query && !is_training) || (config->single_ep_training && is_training)) {
            search_layer(config, query.second, path, entry_points, 1, layer, is_querying);
        } else {
            search_layer(config, query.second, path, entry_points, config->ef_search_upper, layer, is_querying);
        }
        if (config->debug_search)
            cout << "Closest point at layer " << layer << " is " << entry_points[0].second << " (" << entry_points[0].first << ")" << endl;
    }

    if (config->debug_query_search_index == query.first) {
        debug_file = new ofstream(config->runs_prefix + "query_search.txt");
    }
    if (config->gt_dist_log)
        log_neighbors = true;
    
    search_layer(config, query.second, path, entry_points, config->ef_search, 0, is_querying, is_training, is_ignoring);
    if (config->print_path_size) {
        total_path_size += path.size();
    }
    if (config->gt_dist_log)
        log_neighbors = false;
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
    // Select closest elements
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
        Edge* current = nullptr;
        for (auto edge : path) {
            if (edge->target == entry_points[i].second) {
                current = edge;
                //cout << "Current edge found for entry point " << i << " with target " << edge->target << endl;  // Debug print
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
                // cout << "Traversing back to previous edge, size: " << size << ", current target: " << current->target << endl;  // Debug print
            }
        }
    }
    vector<Edge*> direct_path_vector(direct_path.begin(), direct_path.end());
    path = direct_path_vector;
}

bool HNSW::should_terminate(Config* config, priority_queue<pair<float, int>>& top_k, pair<float, int>& top_1, float close_squared, float far_squared, float far_extension_sqaured, bool is_querying, int layer_num) {
    // Compare closest distance to thresholds
    bool beam_width_1 = close_squared > far_squared;
    bool beam_width_2 =  false;
    bool alpha_distance_1 = false;
    bool alpha_distance_2 = false;
    
    if (is_querying && layer_num == 0 && (config->combined_termination || config->use_distance_termination)) {
        float close = sqrt(close_squared);
        float threshold = 2 * sqrt(top_k.top().first) + sqrt(top_1.first);
        alpha_distance_1 = top_k.size() >= config->num_return && close > config->termination_alpha * threshold;
        if (config->use_latest && config->use_break) {
            alpha_distance_2 = top_k.size() >= config->num_return && close > config->termination_alpha * config->break_value * threshold;
            if(far_extension_sqaured > 0 ) 
                beam_width_2 = close_squared > far_extension_sqaured;
        }
    }

    // Return whether closest is too far
    if (!is_querying || layer_num > 0) {
        return beam_width_1;
    } else if (config->combined_termination && config->use_latest) {
        return (alpha_distance_1 && beam_width_1) || alpha_distance_2 || beam_width_2;
    } else if (config->combined_termination) {
        return alpha_distance_1 || beam_width_1;
    } else if (config->use_distance_termination) {
        return alpha_distance_1;
    } else {
        return beam_width_1;
    }
}

void HNSW::search_queries(Config* config, float** queries) {
    ofstream* export_file = NULL;
    if (config->export_queries)
        export_file = new ofstream(config->runs_prefix + "queries.txt");
    
    ofstream* indiv_file = NULL;
    if (config->export_indiv)
        indiv_file = new ofstream(config->runs_prefix + "indiv.txt");

    if (config->gt_dist_log)
        when_neigh_found_file = new ofstream(config->runs_prefix + "when_neigh_found.txt");

    bool use_groundtruth = config->groundtruth_file != "";
    if (use_groundtruth && config->query_file == "") {
        cout << "Warning: Groundtruth file will not be used because queries were generated" << endl;
        use_groundtruth = false;
    }

    vector<vector<int>> actual_neighbors;
    if (use_groundtruth)
        load_ivecs(config->groundtruth_file, actual_neighbors, config->num_queries, config->num_return);
    else
        actual_neighbors.resize(config->num_queries);

    int total_found = 0;
    for (int i = 0; i < config->num_queries; ++i) {
        pair<int, float*> query = make_pair(i, queries[i]);
        if ((config->print_actual || config->print_indiv_found || config->print_total_found || config->export_indiv
            || config->gt_dist_log) && !use_groundtruth) {
            // Get actual nearest neighbors
            priority_queue<pair<float, int>> pq;
            for (int j = 0; j < config->num_nodes; ++j) {
                float dist = calculate_l2_sq(query.second, nodes[j], num_dimensions);
                pq.emplace(dist, j);
                if (pq.size() > config->num_return)
                    pq.pop();
            }

            // Place actual nearest neighbors
            actual_neighbors[i].resize(config->num_return);

            int idx = config->num_return;
            while (idx > 0) {
                --idx;
                actual_neighbors[i][idx] = pq.top().second;
                pq.pop();
            }
        }
        cur_groundtruth = actual_neighbors[i];
        reset_statistics();
        vector<Edge*> path;
        vector<pair<float, int>> found = nn_search(config, path, query, config->num_return);
        if (config->gt_dist_log)
            *when_neigh_found_file << endl;
        
        if (config->print_results) {
            // Print out found
            cout << "Found " << found.size() << " nearest neighbors of [" << query.second[0];
            for (int dim = 1; dim < num_dimensions; ++dim)
                cout << " " << query.second[dim];
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
            cout << "Actual " << config->num_return << " nearest neighbors of [" << query.second[0];
            for (int dim = 1; dim < num_dimensions; ++dim)
                cout << " " << query.second[dim];
            cout << "] : ";
            for (int index : actual_neighbors[i])
                cout << index << " ";
            cout << endl;
        }

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

        if (config->export_queries) {
            *export_file << "Query " << i << endl << query.second[0];
            for (int dim = 1; dim < num_dimensions; ++dim)
                *export_file << "," << query.second[dim];
            *export_file << endl;
            for (auto n_pair : found)
                *export_file << n_pair.second << ",";
            *export_file << endl;
            for (Edge* edge : path) {
                *export_file << edge->target << ",";
            }
            *export_file << endl;
        }
    }

    if (config->gt_dist_log) {
        cout << "Total neighbors found (gt comparison): " << correct_nn_found << " (" << correct_nn_found / (double)(config->num_queries * config->num_return) * 100 << "%)" << endl;
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

    if (config->gt_dist_log) {
        when_neigh_found_file->close();
        delete when_neigh_found_file;
        cout << "Exported when neighbors were found to " << config->runs_prefix << "when_neigh_found.txt" << endl;
    }
}

vector<Edge*> HNSW::get_layer_edges(Config* config, int layer) {
    vector<Edge*> edges;
    for (int i = 0; i < config->num_nodes; i++) {
        // Check if node in adjacency list has at least 'layer' layers
        if (mappings[i].size() - 1 >= layer) {
            for (int j = 0; j < mappings[i][layer].size(); j++) {
                edges.push_back(&mappings[i][layer][j]);
            }
        }
    }
    return edges;
}

HNSW* init_hnsw(Config* config, float** nodes) {
    HNSW* hnsw = new HNSW(config, nodes);
    hnsw->mappings.resize(hnsw->num_nodes);

    // Insert first node into first layer with empty connections vector
    hnsw->num_layers = 1;
    hnsw->mappings[0].resize(1);
    hnsw->entry_point = 0;
    return hnsw;
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

float calculate_l2_sq(HNSW* hnsw, int layer, float* a, float* b, int size) {
    if (layer == 0)
        ++hnsw->layer0_dist_comps;
    else if (layer > 0)
        ++hnsw->upper_dist_comps;
    return calculate_l2_sq(a, b, size);
}

float calculate_l2_sq(float* a, float* b, int size) {
    int parts = size / 8;

    // Initialize result to 0
    __m256 result = _mm256_setzero_ps();

    // Process 8 floats at a time
    for (size_t i = 0; i < parts; ++i) {
        // Load vectors from memory into AVX registers
        __m256 vec_a = _mm256_loadu_ps(&a[i * 8]);
        __m256 vec_b = _mm256_loadu_ps(&b[i * 8]);

        // Compute differences and square
        __m256 diff = _mm256_sub_ps(vec_a, vec_b);
        __m256 diff_sq = _mm256_mul_ps(diff, diff);

        result = _mm256_add_ps(result, diff_sq);
    }

    // Process remaining floats
    float remainder = 0;
    for (size_t i = parts * 8; i < size; ++i) {
        float diff = a[i] - b[i];
        remainder += diff * diff;
    }

    // Sum all floats in result
    float sum[8];
    _mm256_storeu_ps(sum, result);
    for (size_t i = 1; i < 8; ++i) {
        sum[0] += sum[i];
    }

    return sum[0] + remainder;
}

void load_fvecs(const string& file, const string& type, float** nodes, int num, int dim, bool has_groundtruth) {
    ifstream f(file, ios::binary | ios::in);
    if (!f) {
        cout << "File " << file << " not found!" << endl;
        exit(-1);
    }
    cout << "Loading " << num << " " << type << " from file " << file << endl;

    // Read dimension
    int read_dim;
    f.read(reinterpret_cast<char*>(&read_dim), 4);
    if (dim != read_dim) {
        cout << "Mismatch between expected and actual dimension: " << dim << " != " << read_dim << endl;
        exit(-1);
    }

    // Check size
    f.seekg(0, ios::end);
    if (num > f.tellg() / (dim * 4 + 4)) {
        cout << "Requested number of " << type << " is greater than number in file: "
            << num << " > " << f.tellg() / (dim * 4 + 4) << endl;
        exit(-1);
    }
    if (type == "nodes" && num != f.tellg() / (dim * 4 + 4) && has_groundtruth) {
        cout << "You must load all " << f.tellg() / (dim * 4 + 4) << " nodes if you want to use a groundtruth file" << endl;
        exit(-1);
    }

    f.seekg(0, ios::beg);
    for (int i = 0; i < num; i++) {
        // Skip dimension size
        f.seekg(4, ios::cur);

        // Read point
        nodes[i] = new float[dim];
        f.read(reinterpret_cast<char*>(nodes[i]), dim * 4);
    }
    f.close();
}

void load_ivecs(const string& file, vector<vector<int>>& results, int num, int num_return) {
    ifstream f(file, ios::binary | ios::in);
    if (!f) {
        cout << "File " << file << " not found!" << endl;
        exit(-1);
    }
    cout << "Loading groundtruth from file " << file << endl;

    // Read width
    int width;
    f.read(reinterpret_cast<char*>(&width), 4);
    if (num_return > width) {
        cout << "Requested num_return is greater than width in file: " << num_return << " > " << width << endl;
        exit(-1);
    }

    // Check size
    f.seekg(0, ios::end);
    if (num > f.tellg() / (width * 4 + 4)) {
        cout << "Requested number of queries is greater than number in file: "
            << num << " > " << f.tellg() / (width * 4 + 4) << endl;
        exit(-1);
    }

    results.reserve(num);
    f.seekg(0, ios::beg);
    for (int i = 0; i < num; i++) {
        // Skip list size
        f.seekg(4, ios::cur);

        // Read point
        int values[num_return];
        f.read(reinterpret_cast<char*>(values), num_return * 4);
        results.push_back(vector<int>(values, values + num_return));

        // Skip remaining values
        f.seekg((width - num_return) * 4, ios::cur);
    }
    f.close();
}

void load_hnsw_files(Config* config, HNSW* hnsw, float** nodes, bool is_benchmarking) {
    // Check file and parameters
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

    int opt_con, max_con, max_con_0, ef_con;
    int num_nodes;
    int num_layers;
    info_file >> opt_con >> max_con >> max_con_0 >> ef_con;
    info_file >> num_nodes;
    info_file >> num_layers;

    // Check if number of nodes match
    if (num_nodes != config->num_nodes) {
        cout << "Mismatch between loaded and expected number of nodes" << endl;
        return;
    }

    // Check if construction parameters match
    if (opt_con != config->optimal_connections || max_con != config->max_connections ||
        max_con_0 != config->max_connections_0 || ef_con != config->ef_construction) {
        cout << "Mismatch between loaded and expected construction parameters" << endl;
        return;
    }

    if (is_benchmarking) {
        long long construct_layer0_dist_comps;
        long long construct_upper_dist_comps;
        double construct_duration;
        info_file >> construct_layer0_dist_comps;
        info_file >> construct_upper_dist_comps;
        info_file >> construct_duration;

        auto start = chrono::high_resolution_clock::now();
        cout << "Loading graph with construction parameters: "
            << config->optimal_connections << ", " << config->max_connections << ", "
            << config->max_connections_0 << ", " << config->ef_construction << endl;
        
        hnsw->num_layers = num_layers;
        load_hnsw_graph(config, hnsw, graph_file, nodes, num_nodes, num_layers);
        
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Load time: " << duration / 1000.0 << " seconds, ";
        cout << "Construction time: " << construct_duration << " seconds, ";
        cout << "Distance computations (layer 0): " << construct_layer0_dist_comps <<", ";
        cout << "Distance computations (top layers): " << construct_upper_dist_comps << endl;
    } else {
        hnsw->num_layers = num_layers;
        load_hnsw_graph(config, hnsw, graph_file, nodes, num_nodes, num_layers);
    }
}

void load_hnsw_graph(Config* config, HNSW* hnsw, ifstream& graph_file, float** nodes, int num_nodes, int num_layers) {
    // Load node neighbors
    for (int i = 0; i < num_nodes; ++i) {
        int layers;
        graph_file.read(reinterpret_cast<char*>(&layers), sizeof(layers));
        hnsw->mappings[i].resize(layers);

        // Load layers
        for (int j = 0; j < layers; ++j) {
            int num_neighbors;
            graph_file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
            hnsw->mappings[i][j].reserve(num_neighbors);

            // Load neighbors
            for (int k = 0; k < num_neighbors; ++k) {
                int index;
                float distance;
                graph_file.read(reinterpret_cast<char*>(&index), sizeof(index));
                graph_file.read(reinterpret_cast<char*>(&distance), sizeof(distance));
                hnsw->mappings[i][j].emplace_back(Edge(index, distance, config->initial_cost, config->initial_benefit));
            }
        }
    }

    // Load entry point
    int entry_point;
    graph_file.read(reinterpret_cast<char*>(&entry_point), sizeof(entry_point));
    hnsw->entry_point = entry_point;
}

void save_hnsw_files(Config* config, HNSW* hnsw, const string& name, long int duration) {
    // Export graph to file
    ofstream graph_file(config->runs_prefix + "graph_" + name + ".bin");

    // Export edges
    for (int i = 0; i < config->num_nodes; ++i) {
        int layers = hnsw->mappings[i].size();

        // Write number of layers
        graph_file.write(reinterpret_cast<const char*>(&layers), sizeof(layers));

        // Write layers
        for (int j = 0; j < layers; ++j) {
            int num_neighbors = hnsw->mappings[i][j].size();

            // Write number of neighbors
            graph_file.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));

            // Write neighbors
            for (int k = 0; k < num_neighbors; ++k) {
                auto n_pair = hnsw->mappings[i][j][k];
                
                // Write index and distance
                graph_file.write(reinterpret_cast<const char*>(&n_pair.target), sizeof(n_pair.target));
                graph_file.write(reinterpret_cast<const char*>(&n_pair.distance), sizeof(n_pair.distance));
            }
        }
    }

    // Save entry point
    graph_file.write(reinterpret_cast<const char*>(&hnsw->entry_point), sizeof(hnsw->entry_point));
    graph_file.close();

    // Export construction parameters
    ofstream info_file(config->runs_prefix + "info_" + name + ".txt");
    info_file << config->optimal_connections << " " << config->max_connections << " "
              << config->max_connections_0 << " " << config->ef_construction << endl;
    info_file << config->num_nodes << endl;
    info_file << hnsw->num_layers << endl;
    info_file << hnsw->layer0_dist_comps << endl;
    info_file << hnsw->upper_dist_comps << endl;
    info_file << duration << endl;

    cout << "Exported graph to " << config->runs_prefix + "graph_" + name + ".bin" << endl;
}

void load_nodes(Config* config, float** nodes) {
    if (config->load_file != "") {
        if (config->load_file.size() >= 6 && config->load_file.substr(config->load_file.size() - 6) == ".fvecs") {
            // Load nodes from fvecs file
            load_fvecs(config->load_file, "nodes", nodes, config->num_nodes, config->dimensions, config->groundtruth_file != "");
            return;
        }
    
        // Load nodes from file
        ifstream f(config->load_file, ios::in);
        if (!f) {
            cout << "File " << config->load_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << config->num_nodes << " nodes from file " << config->load_file << endl;

        for (int i = 0; i < config->num_nodes; i++) {
            nodes[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> nodes[i][j];
            }
        }

        f.close();
        return;
    }

    cout << "Generating " << config->num_nodes << " random nodes" << endl;

    mt19937 gen(config->graph_seed);
    uniform_real_distribution<float> dis(config->gen_min, config->gen_max);

    for (int i = 0; i < config->num_nodes; i++) {
        nodes[i] = new float[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            nodes[i][j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
    }
}

void load_queries(Config* config, float** nodes, float** queries) {
    mt19937 gen(config->query_seed);
    if (config->query_file != "") {
        if (config->query_file.size() >= 6 && config->query_file.substr(config->query_file.size() - 6) == ".fvecs") {
            // Load queries from fvecs file
            load_fvecs(config->query_file, "queries", queries, config->num_queries, config->dimensions, config->groundtruth_file != "");
            return;
        }

        // Load queries from file
        ifstream f(config->query_file, ios::in);
        if (!f) {
            cout << "File " << config->query_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << config->num_queries << " queries from file " << config->query_file << endl;

        for (int i = 0; i < config->num_queries; i++) {
            queries[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> queries[i][j];
            }
        }

        f.close();
        return;
    }

    if (config->load_file == "") {
        // Generate random queries (same as get_nodes)
        cout << "Generating " << config->num_queries << " random queries" << endl;
        uniform_real_distribution<float> dis(config->gen_min, config->gen_max);

        for (int i = 0; i < config->num_queries; i++) {
            queries[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                queries[i][j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
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
    for (int i = 0; i < config->num_queries; i++) {
        queries[i] = new float[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            queries[i][j] = round(dis_array[j](gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
    }

    delete[] lower_bound;
    delete[] upper_bound;
    delete[] dis_array;
}
