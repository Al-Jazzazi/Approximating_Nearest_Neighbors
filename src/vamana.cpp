#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <queue>
#include <unordered_set>
#include <immintrin.h>
#include "vamana.h"


/// reduce K_QUERY to 1, 50, etc -> find out what's the issue
/// construction L and query L are different -> try diff vals
/// L should not be related to K -> try to find a good number (for both L)
/// first fix construction L to a good number and then for query L
/// try max outedge != K -> check what they do in paper
/// 2 round vamana
/// try different / larger dataset

using namespace std;


int distanceCalculationCount = 0;
int alpha = 1.2;
int K = 30; // Num of NNs when building Vamana graph
// int K_QUERY = 100; // Num of NNs found for each query
// int K_TRUTH = 100; // Num of NNs provided by ground truth for each query
int R = 50; // Max outedge


int main() {
    // Construct Vamana index
    Config* config = new Config();
    auto start = std::chrono::high_resolution_clock::now();
    Graph G = Vamana(config, alpha, K, R);
    auto end = chrono::high_resolution_clock::now();;
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Load time: " << duration / 1000.0 << " seconds, ";
    // print_100_mappings(G, config);
    
    float** queries = new float*[config->num_queries];
    load_queries(config, G.nodes, queries);
   
    // if(config->export_graph)
    //     G.to_files(config,"vamana_1M");

    // Search queries
    runQueries(config, G, queries);


  
    // Clean up
    for (int i = 0; i < config->num_queries; ++i)
        delete[] queries[i];
    delete[] queries;

    delete config;
}



ostream& operator<<(ostream& os, const Graph& rhs) {
    for (int i = 0; i < rhs.num_nodes; i++) {
        cout << i << " : ";
        for (int neighbor : rhs.mappings[i]) {
            cout << neighbor << " ";
        }
        cout << endl;
    }
    return os;
}

Graph::Graph(Config* config) {
    num_nodes = config->num_nodes;
    DIMENSION = config->dimensions;
    nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    mappings.resize(num_nodes);
    for (int i = 0; i < mappings.size(); i++) {
        mappings[i] = {};
    }
    int start = -1;
}

Graph::~Graph() {
    for (int i = 0; i < num_nodes; i++) {
        delete[] nodes[i];
    }
    delete[] nodes;
}

void Graph::to_files(Config* config, const string& graph_name) {
    // Export graph to file
    ofstream graph_file(config->loaded_graph_file);

    // Export edges
    for (int i = 0; i < num_nodes; ++i) {
        // Write number of neighbors
        int num_neighbors = mappings[i].size();
        graph_file.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));

        // Write index and distance of each neighbor
        for (int neighbor : mappings[i]) {
            graph_file.write(reinterpret_cast<const char*>(&neighbor), sizeof(neighbor));
        }
    }
    graph_file.close();
    cout << "Exported graph to " << config->runs_prefix + "graph_" + graph_name + ".bin" << endl;

}

void Graph::from_files(Config* config, bool is_benchmarking) {
    // Open files
    ifstream graph_file(config->loaded_graph_file);
    cout << "Loading saved graph from " << config->loaded_graph_file << endl;
    if (!graph_file) {
        cout << "File " << config->loaded_graph_file << " not found!" << endl;
        return;
    }

    // Process graph file
    for (int i = 0; i < num_nodes; ++i) {
        int num_neighbors;
        graph_file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
        // Load each neighbor
        
        for (int j = 0; j < num_neighbors; ++j) {
            int neighbor;
            graph_file.read(reinterpret_cast<char*>(&neighbor), sizeof(neighbor));
            mappings[i].emplace(neighbor);
        }
        if(i%10000 == 0)
            cout << i << endl;
            cout << "num n" << num_neighbors <<endl;
    }

    cout << "done with loading"  <<endl;
}


void Graph::randomize(int R) {
    for (int i = 0; i < num_nodes; i++) {
        set<int> neighbors = {};
        for (int j = 0; j < R; j++) {
            int random = rand() % num_nodes; // find a random node
            while (random == i) random = rand() % num_nodes;
            neighbors.insert(random);
        }
        mappings[i] = neighbors;
    }
}



float Graph::findDistance(int i, float* query) const {
    distanceCalculationCount++;
    return calculate_l2_sq(nodes[i], query, DIMENSION);
}


float Graph::findDistance(int i, int j) const {
    distanceCalculationCount++;
    return calculate_l2_sq(nodes[i], nodes[j], DIMENSION);
}


// void Graph::sanityCheck(Config* config, const vector<vector<int>>& allResults) const {
//     // vector<set<int>> gives all NNs
//     fstream groundTruth;
//     groundTruth.open(config->groundtruth_file);
//     if (!groundTruth) {cout << "Ground truth file not open" << endl;}
//     int each;
//     float totalCorrect = 0;
//     float result;
//     for (int j = 0; j < config->num_queries; j++) {
//         int correct = 0;
//         vector<int> allTruths = {};
//         for (int i = 0; i < K_TRUTH; i++) {
//             groundTruth >> each;
//             allTruths.push_back(each);
//         }
//         vector<int> eachResult = {};
//         for (int count = 0; count < K_QUERY; count++) {
//             for (int ea : allResults[j]) {
//                 if (allTruths[count] == ea) correct++;
//             }
//         }
//         result = correct * 100 / K_QUERY;
//         totalCorrect += result;
//         cout << "Found " << result << "% among " << K_QUERY << " closest neighbors" << endl;
//     }
//     result = totalCorrect / config->num_queries;
//     cout << "Average correctness: " << result << '%' << endl;
// }

 void Graph:: query(Config* config, int start, vector<vector<int>>& allResults, float** queries) {
   
    for (int k = 0; k < config->num_queries; k++) {
        if (k % 1000 == 0) cout << "Processing " << k << endl;
        float* thisQuery = queries[k];
        // cout << "flag 1" << endl;
        // auto startTime = std::chrono::high_resolution_clock::now();
        // vector<int> result = GreedySearch(*this, start, thisQuery, L_QUERY);
        // auto endTime = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        vector<int> result;
        BeamSearch(*this,config, start, thisQuery, config->ef_search, result);
        allResults.push_back(result);
    }
    cout << "All queries processed" << endl;
}


// void Graph::queryTest(int start) {
//     vector<float*> queryNodes = {};
//     int queryCount = 0;
//     int correct = 0;
//     while(queryCount < 100) {
//         int random = rand() % num_nodes;
//         queryNodes.push_back(nodes[random]);
//         queryCount++;
//     }
//     for (float* each : queryNodes) {
//         auto startTime = std::chrono::high_resolution_clock::now();
//         vector<int> result = GreedySearch(*this, start, each, L_QUERY);
//         auto endTime = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//         int closestNode = 0;
//         float shortestDistance = findDistance(0, each);
//         for (int i = 0; i < num_nodes; i++) {
//             float distance = findDistance(i, each);
//             if (distance < shortestDistance) {
//                 closestNode = i;
//                 shortestDistance = distance;
//             }
//         }
//         for (auto i : result) {
//             if (i == closestNode) correct++;
//         }
//     }
//     cout << "Total correct number: " << correct << endl;
// }

// void Graph::queryBruteForce(Config* config, int start) {
//     fstream f;
//     f.open(config->query_file);
//     if (!f) {cout << "Query file not open" << endl;}
//     float** queries = new float*[config->num_queries];
//     float each;
//     for (int i = 0; i < config->num_queries; i++) {
//         queries[i] = new float[DIMENSION];
//         for (int j = 0; j < DIMENSION; j++) {
//             f >> each;
//             queries[i][j] = each;
//         }
//     }
//     cout << "All queries read" << endl;   
//     fstream groundTruth;
//     groundTruth.open(config->groundtruth_file);
//     if (!groundTruth) {cout << "Ground truth file not open" << endl;}
//     int totalCorrect = 0;
//     float result;
//     for (int j = 0; j < config->num_queries; j++) {
//         vector<int> allTruths = {};
//         for (int i = 0; i < K_TRUTH; i++) {
//             groundTruth >> each;
//             allTruths.push_back(each);
//         }
//         float* query = queries[j];
//         int closest = 0;
//         double closestDist = findDistance(closest, query);
//         for (int k = 0; k < num_nodes; k++) {
//             double newDist = findDistance(k, query);
//             if (newDist < closestDist) {
//                 closest = k;
//                 closestDist = newDist;
//             }
//         }
//         if (allTruths[0] == closest) {
//             totalCorrect++;
//         } else {
//             cout << allTruths[0] << ' ' << closest << endl;
//         }
//     }
//     result = totalCorrect;
//     cout << "Average correctness: " << result << '%' << endl;
// }


void runQueries(Config* config, Graph& graph, float** queries){
    int start= findStart(config, graph);
    vector<vector<int>> results;
    graph.query(config, start, results, queries);
    vector<vector<int>> actualResults;
    get_actual_neighbors(config, actualResults, graph.nodes, queries);
    int similar =0 ;
    cout << "results.size() " << results.size() <<  ", actualResults.size() " << actualResults.size() << endl ; 
    cout << "results[0].size() " << results[0].size() <<  ", actualResults[0].size() " << actualResults[0].size() << endl ; 
    cout << "results[0] " << results[0][0] <<  ", actualResults[0] " << actualResults[0][0] << endl ; 
    cout << "results[0] " << graph.findDistance(results[0][0], queries[0]) <<  ", actualResults[0] " << graph.findDistance(actualResults[0][0], queries[0]) << endl ; 

    for (int j = 0; j < config->num_queries; ++j) {
                // Find similar neighbors
                unordered_set<int> actual_set(actualResults[j].begin(), actualResults[j].end());
                unordered_set<int> intersection;
                // float actual_gain = 0;
                // float ideal_gain = 0;
                
              
                for (int k = 0; k < results[j].size(); ++k) {
                    auto n_pair = results[j][k];
                    // float gain = 1 / log2(k + 2);
                    // ideal_gain += gain;
                    if (actual_set.find(n_pair) != actual_set.end()) {
                        intersection.insert(n_pair);
                        // actual_gain += gain;
                    }
                }
                similar += intersection.size();
    
        }
    cout << "similar" << similar << endl; 
    double recall = (double) similar / (config->num_queries * config->num_return);
    cout << "Recall of Vamana is " << recall;
}


void get_actual_neighbors(Config* config, vector<vector<int>>& actual_neighbors, float** nodes, float** queries) {
    bool use_groundtruth = config->groundtruth_file != "";
    if (use_groundtruth && config->query_file == "") {
        cout << "Warning: Groundtruth file will not be used because queries were generated" << endl;
        use_groundtruth = false;
    }
    if (use_groundtruth) {
        // Load actual nearest neighbors
        load_ivecs(config->groundtruth_file, actual_neighbors, config->num_queries, config->num_return);
    } else {
        // Calcuate actual nearest neighbors
        auto start = chrono::high_resolution_clock::now();
        knn_search(config, actual_neighbors, nodes, queries);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Brute force time: " << duration / 1000.0 << " seconds" << endl;
    }
 
}
/// L, V, diff between L and V
/// L -> priority queue with distance and index
/// V -> vector with inde
/// diff -> priority queue with distance and index -> 
vector<int> GreedySearch(Graph& graph, int start,  float* query, int L) {    
    vector<int> result;
    priority_queue<tuple<float, int>> List; // max priority queue
    set<int> ListSet = {};
    float distance = graph.findDistance(start,  query);
    List.push({distance, start}); // L <- {s}
    ListSet.insert(start);
    vector<int> Visited = {};
    priority_queue<tuple<float, int>> diff; // min priority queue
    diff.push({-1 * distance, start});

    while (diff.size() != 0) {
        tuple<float, int> top = diff.top(); // get the best candidate
        Visited.push_back(get<1>(top));
        for (int j : graph.mappings[get<1>(top)]) {
            float dist = graph.findDistance(j, query);
            tuple<float, int> newNode = {dist, j};
            bool inserted = false;
           
            if (ListSet.find(j) != ListSet.end()) {inserted = true;}
            
            if (!inserted) List.push(newNode);
            ListSet.insert(j);
        }

        while (List.size() > L) List.pop();

        priority_queue<tuple<float, int>> copy = List;
        diff = {};
        while (copy.size() != 0) {
            tuple<float, int> next = copy.top();
            copy.pop();
            bool exists = false;
            for (int k : Visited) {
                if (k == get<1>(next)) exists = true;
            }
            if (!exists) diff.push({-1 * get<0>(next), get<1>(next)});
        }
    }
    while (List.size() != 0) {
        result.push_back(get<1>(List.top()));
        List.pop();
    }
    return result;
}



void BeamSearch(Graph& graph, Config* config,int start,  float* query, int bw, vector<int>& closest){
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> candidates;
    unordered_set<int> visited;
    priority_queue<pair<float, int>> found;
    // cout << "flag 2 " << endl;
    //priority_queue<pair<float, int>> top_k;
    float distance = graph.findDistance(start, query);
    candidates.emplace(make_pair(distance,start));
    visited.emplace(start);
    found.emplace(make_pair(distance,start));


    int candidates_popped_per_q = 0;
    int iteration = 0;
    // float far_dist = found.top().first;
    while (!candidates.empty()) {
        // cout << "flag 3 " << endl;
        float far_dist = found.top().first;
        int closest = candidates.top().second;
        float close_dist = candidates.top().first;
        candidates.pop();

         if (close_dist > far_dist) {
            break;
        }
        set<int>& neighbors = graph.mappings[start];
        for (int neighbor : neighbors) {
            if(visited.find(neighbor) == visited.end())
                visited.insert(neighbor);
            // cout << "flag 4 " << endl;
            float far_inner_dist = found.top().first;
            float neighbor_dist = graph.findDistance(neighbor,query);
            if (neighbor_dist < far_inner_dist || found.size() < bw) {
                candidates.emplace(make_pair(neighbor_dist, neighbor));
                found.emplace(neighbor_dist, neighbor);
                if (found.size() > bw){
                    found.pop();
                }
            }

        }

    }
    // cout << "flag 5" << endl;
    int idx =found.size(); 
    closest.clear();
    closest.resize(idx);
    while (idx > 0) {
        --idx;
        closest[idx] = found.top().second;
        found.pop();

    }


    closest.resize(min(closest.size(), (size_t)config->num_return));
    // cout << "flag 6 " << endl;

}

void RobustPrune(Graph& graph, int point, vector<int>& candidates, long threshold, int R) {
    set<int> neighbors = graph.mappings[point];
    for (int i : neighbors) {
        candidates.push_back(i);
    }
    for (int j = 0; j < candidates.size(); j++) {
        if (candidates[j] == point) {
            candidates[j] = candidates[candidates.size()-1];
            candidates.pop_back();
            break;
        }
    }
    graph.mappings[point] = {};

    while (candidates.size() != 0) {
        // find p* <- closest neighbor to p
        int bestCandidate = *candidates.begin();
        for (int j : candidates) {
            if (graph.findDistance(j, graph.nodes[point]) < graph.findDistance(bestCandidate, graph.nodes[point])) {
                bestCandidate = j;
            }
        }
        for (int j = 0; j < candidates.size(); j++) {
            if (candidates[j] == bestCandidate) {
                candidates[j] = candidates[candidates.size()-1];
                candidates.pop_back();
                break;
            }
        }
        // add best candidate back to p's neighborhood
        set<int> edges = graph.mappings[point];
        edges.insert(bestCandidate);
        graph.mappings[point] = edges;
        // neighborhood is full
        if (graph.mappings[point].size() == R) {
            break;
        }
        vector<int> copy;
        for (int k : candidates) {
            if (graph.findDistance(point, graph.nodes[k]) < threshold * graph.findDistance(bestCandidate, graph.nodes[k])) {
                copy.push_back(k);
            }
        }
        candidates = copy;
    }
}
//this returns point closest to centroid 

//return find start
int findStart(Config* config, const Graph& g) {
    float* center = new float[g.DIMENSION];
    for(int k = 0; k < g.DIMENSION; k++){
        center[k] = 0; 
    }
    for (int j = 0; j < g.num_nodes; j++) {
        for (int k = 0; k < g.DIMENSION; k++) {
            center[k] += g.nodes[j][k];
        }
    }
    for (int i = 0; i < g.DIMENSION; i++) {
        center[i] /= g.num_nodes;
    }
    int closest = 0;
    float closest_dist = MAXFLOAT;
    for (int m = 0; m < g.num_nodes; m++) {
        float this_dist = g.findDistance(m, center);
        if (this_dist < closest_dist) {
            closest_dist = this_dist;
            closest = m;
        }
    }
    return closest;
}

Graph Vamana(Config* config, long alpha, int L, int R) {
    Graph graph(config);
    if(config->load_graph_file){
        graph.from_files(config, config->export_benchmark);
        print_100_nodes(graph, config );
        return graph;
    }

    cout << "Start of Vamana" << endl;
    cout << "Randomizing edges" << endl;
    graph.randomize(R);
    cout << "Randomized edges" << endl;
    // print_100_mappings(graph, config);
    cout << "Random graph: " << endl;
    int s = findStart(config, graph);
    cout << "The centroid is #" << s << endl;
    for (int i = 0; i < 2; i++) {
        long actual_alpha = (i == 0) ? 1 : alpha;
        vector<int> sigma;
        for (int i = 0; i < config->num_nodes; i++) {
            sigma.push_back(i);
        }
        // unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        shuffle(sigma.begin(), sigma.end(), default_random_engine(config->shuffle_seed));
        int count = 0;
        for (int i : sigma) {
            if (count % 1000 == 0) cout << "Num of node processed: " << count << endl;
            count++;
            vector<int> result = GreedySearch(graph, s, graph.nodes[i], config->ef_construction);
            RobustPrune(graph, i, result, actual_alpha, R);
            set<int> neighbors = graph.mappings[i];
            for (int j : neighbors) {
                set<int> unionV = graph.mappings[j]; 
                vector<int> unionVec;

                for (int i : unionV) {
                    unionVec.push_back(i);
                }
                unionV.insert(i);
                unionVec.push_back(i);
                if (unionV.size() > R) {
                    RobustPrune(graph, j, unionVec, actual_alpha, R);
                } else {
                    graph.mappings[j]= unionV;
                }
            }
        }
    }
    cout << "End of Vamana" << endl;
    return graph;
}

void print_100_nodes(const Graph& graph, Config* config){
    for(int i=0; i<config->num_nodes; i++){
        if(i ==100 ) break;
        cout << "i: " << i << endl;
        for(int j =0; j< graph.DIMENSION; j++){
            cout << ", " << graph.nodes[i][j]; 
        }
        cout <<endl;
    }
}

void print_100_mappings(const Graph& graph, Config* config){
    for(int i = 0; i<graph.mappings.size(); i++){
        if(i ==100 ) break;
        cout << "i: " <<graph.mappings[i].size() <<endl;
       
        
        
    }
}

