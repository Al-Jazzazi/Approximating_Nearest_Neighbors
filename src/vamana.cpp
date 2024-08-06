#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <queue>
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
int K_QUERY = 100; // Num of NNs found for each query
int K_TRUTH = 100; // Num of NNs provided by ground truth for each query
int R = 50; // Max outedge
int L = 100; // beam search width
int L_QUERY = 100;

int main() {
    // Construct Vamana index
    Config* config = new Config();
    auto start = std::chrono::high_resolution_clock::now();
    float** nodes; 
    Graph G = Vamana(config, alpha, K, R);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Search queries
    size_t entry = findStart(config, G);
    G.queryTest(entry);
    start = std::chrono::high_resolution_clock::now();
    distanceCalculationCount = 0;
    vector<vector<size_t>> allResults = G.query(config, entry);
    stop = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // G.sanityCheck(config->groundtruth_file, allResults);
    std::cout << "Duration of Vamana: "<< duration.count()/1000 << " millisecond(s)" << endl;
    cout << "Duration of Each Query: "<< duration2.count()/1000/config->num_queries << " millisecond(s)"<< endl;
    cout << "Number of distance calculation per query: " << distanceCalculationCount/config->num_queries << endl;

    // Clean up
    delete config;
}

// ostream& operator<<(ostream& os, const DataNode& rhs) {
//     for (size_t i = 0; i < DIMENSION; i++) {
//         os << rhs.coordinates[i] << ' ';
//     }
//     os << endl;
//     return os;
// }

// bool operator==(const DataNode& lhs, const DataNode& rhs) {
//     if (lhs.dimension != rhs.dimension) return false;
//     for (size_t ind = 0; ind < lhs.dimension; ind++) {
//         if (lhs.coordinates[ind] != rhs.coordinates[ind]) return false;
//     }
//     return true;
// }

// DataNode::DataNode() {}
// DataNode::DataNode(double* coord) {
//     dimension = DIMENSION;
//     coordinates = coord;
// }
// void DataNode::setWord(const string& theWord) {
//     word = theWord;
// }
//void DataNode::sumArraysAVX(int* array1, int* array2, int* result, int size) const {
////        cout << "In sumArraysAVX" << endl;
//    int vectorSize = sizeof(__m256) / sizeof(float);
//    for (int i = 0; i < size; i += vectorSize) {
//        __m256 vec1 = _mm256_load_ps(&array1[i]);
//        __m256 vec2 = _mm256_load_ps(&array2[i]);
//        __m256 sub = _mm256_sub_ps(vec1, vec2);
//        _mm256_store_ps(&result[i], sub);
//    }
//}
//long long int DataNode::findDistanceAVX(const DataNode& other) const {
//    float distance = 0;
////        cout << "In findDistance" << endl;
//    if (dimension == other.dimension) {
//        void* ptr = aligned_alloc(32, DIMENSION*8);
//        float* result = new(ptr) float[DIMENSION];
//        sumArraysAVX(coordinates, other.coordinates, result, DIMENSION);
//        for (size_t i = 0; i < DIMENSION; i++) {
//            if (result[i] < 0) result[i] *= -1;
//            distance += result[i];
//        }
//    }
//    return distance;
//}


// double DataNode::findDistance(const DataNode& other) const {
//     distanceCalculationCount++;
//     double distance = 0;
// //    cout << "In findDistance ";
//     if (dimension == other.dimension) {
//         for (size_t i = 0; i < DIMENSION; i++) {
//             double result = coordinates[i] - other.coordinates[i];
//             result *= result;
//             distance += result;
//         }
//     }
//     return sqrt(distance);
// }

// bool DataNode::compare(double* coord) const {
//     for (size_t i = 0; i < DIMENSION; i++) {
//         if (coord[i] != coordinates[i]) return false;
//     }
//     return true;
// }


ostream& operator<<(ostream& os, const Graph& rhs) {
    for (size_t i = 0; i < rhs.num_nodes; i++) {
        cout << i << " : ";
        for (size_t neighbor : rhs.mappings[i]) {
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
}

Graph::~Graph() {
    for (int i = 0; i < num_nodes; i++) {
        delete[] nodes[i];
    }
    delete[] nodes;
}

void Graph::to_files(Config* config, const string& graph_name) {
    // Export graph to file
    ofstream graph_file(config->runs_prefix + "graph_" + graph_name + ".bin");

    // Export edges
    for (size_t i = 0; i < num_nodes; ++i) {
        // Write number of neighbors
        int num_neighbors = mappings[i].size();
        graph_file.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));

        // Write index and distance of each neighbor
        for (size_t neighbor : mappings[i]) {
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
            mappings[i].insert(neighbor);
        }
    }
}

void Graph::randomize(int R) {
    for (size_t i = 0; i < num_nodes; i++) {
        set<size_t> neighbors = {};
        for (size_t j = 0; j < R; j++) {
            size_t random = rand() % num_nodes; // find a random node
            while (random == i) random = rand() % num_nodes;
            neighbors.insert(random);
        }
        mappings[i] = neighbors;
    }
}



float Graph::findDistance(size_t i, float* query) const {
    distanceCalculationCount++;
    return calculate_l2_sq(nodes[i], query, DIMENSION);
}



void Graph::sanityCheck(Config* config, const vector<vector<size_t>>& allResults) const {
    // vector<set<size_t>> gives all NNs
    fstream groundTruth;
    groundTruth.open(config->groundtruth_file);
    if (!groundTruth) {cout << "Ground truth file not open" << endl;}
    int each;
    float totalCorrect = 0;
    float result;
    for (size_t j = 0; j < config->num_queries; j++) {
        int correct = 0;
        vector<size_t> allTruths = {};
//        cout << "Ground truths: ";
        for (size_t i = 0; i < K_TRUTH; i++) {
            groundTruth >> each;
            allTruths.push_back(each);
//            cout << each << ' ';
        }
//        cout << endl;
        vector<size_t> eachResult = {};
        for (int count = 0; count < K_QUERY; count++) {
            for (size_t ea : allResults[j]) {
                if (allTruths[count] == ea) correct++;
            }
        }
        result = correct * 100 / K_QUERY;
        totalCorrect += result;
        cout << "Found " << result << "% among " << K_QUERY << " closest neighbors" << endl;
    }
    result = totalCorrect / config->num_queries;
    cout << "Average correctness: " << result << '%' << endl;
}

vector<vector<size_t>> Graph::query(Config* config, size_t start) {
    fstream f;
    f.open(config->query_file);
    if (!f) {cout << "Query file not open" << endl;}
    float** queries = new float*[config->num_queries];
    double each;
    for (size_t i = 0; i < config->num_queries; i++) {
        //void* ptr = aligned_alloc(32, DIMENSION*8);
        //float* coord = new(ptr) float[DIMENSION];
        queries[i] = new float[DIMENSION];
        for (size_t j = 0; j < DIMENSION; j++) {
            f >> each;
            queries[i][j] = each;
        }
    }
    cout << "All queries read" << endl;
    vector<vector<size_t>> allResults = {};
    for (size_t k = 0; k < config->num_queries; k++) {
        if (k % 1000 == 0) cout << "Processing " << k << endl;
        float* thisQuery = queries[k];
        auto startTime = std::chrono::high_resolution_clock::now();
        vector<size_t> result = GreedySearch(*this, start, thisQuery, L_QUERY);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        allResults.push_back(result);
    }
    cout << "All queries processed" << endl;
    return allResults;
}


void Graph::queryTest(size_t start) {
    vector<float*> queryNodes = {};
    int queryCount = 0;
    size_t correct = 0;
    while(queryCount < 100) {
        size_t random = rand() % num_nodes;
        queryNodes.push_back(nodes[random]);
        queryCount++;
    }
    for (float* each : queryNodes) {

        auto startTime = std::chrono::high_resolution_clock::now();
        vector<size_t> result = GreedySearch(*this, start, each, L_QUERY);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        size_t closestNode = 0;
        float shortestDistance = findDistance(0, each);
        for (size_t i = 0; i < num_nodes; i++) {
            float distance = findDistance(i, each);
            if (distance < shortestDistance) {
                closestNode = i;
                shortestDistance = distance;
            }
        }
        // cout << closestNode << ' ' << result[0] << endl;
        for (auto i : result) {
            if (i == closestNode) correct++;
        }
    }
    cout << "Total correct number: " << correct << endl;
}

void Graph::queryBruteForce(Config* config, size_t start) {
    fstream f;
    f.open(config->query_file);
    if (!f) {cout << "Query file not open" << endl;}
    float** queries = new float*[config->num_queries];
    float each;
    for (size_t i = 0; i < config->num_queries; i++) {
        //void* ptr = aligned_alloc(32, DIMENSION*8);
        //float* coord = new(ptr) float[DIMENSION];
        queries[i] = new float[DIMENSION];
        for (size_t j = 0; j < DIMENSION; j++) {
            f >> each;
            queries[i][j] = each;
        }
    }
    cout << "All queries read" << endl;
    
    fstream groundTruth;
    groundTruth.open(config->groundtruth_file);
    if (!groundTruth) {cout << "Ground truth file not open" << endl;}
    int totalCorrect = 0;
    float result;
    for (size_t j = 0; j < config->num_queries; j++) {
        vector<size_t> allTruths = {};
        for (size_t i = 0; i < K_TRUTH; i++) {
            groundTruth >> each;
            allTruths.push_back(each);
        }
        float* query = queries[j];
        size_t closest = 0;
        double closestDist = findDistance(closest, query);
        for (size_t k = 0; k < num_nodes; k++) {
            double newDist = findDistance(k, query);
            if (newDist < closestDist) {
                closest = k;
                closestDist = newDist;
            }
        }
        if (allTruths[0] == closest) {
            totalCorrect++;
        } else {
            cout << allTruths[0] << ' ' << closest << endl;
        }
    }
    result = totalCorrect;
    cout << "Average correctness: " << result << '%' << endl;
}

/// L, V, diff between L and V
/// L -> priority queue with distance and index
/// V -> vector with inde
/// diff -> priority queue with distance and index -> 
vector<size_t> GreedySearch(Graph& graph, size_t start,  float* query, size_t L) {    
    vector<size_t> result;
    priority_queue<tuple<float, size_t>> List; // max priority queue
    set<size_t> ListSet = {};
    float distance = graph.findDistance(start,  query);
    List.push({distance, start}); // L <- {s}
    ListSet.insert(start);
    vector<size_t> Visited = {};
    priority_queue<tuple<float, size_t>> diff; // min priority queue
    diff.push({-1 * distance, start});
    while (diff.size() != 0) {
        tuple<float, size_t> top = diff.top(); // get the best candidate
        Visited.push_back(get<1>(top));
        for (size_t j : graph.mappings[get<1>(top)]) {
            float dist = graph.findDistance(j, query);
            tuple<float, size_t> newNode = {dist, j};
            bool inserted = false;
            for (size_t i : ListSet) {
                if (i == j) {inserted = true;}
            }
            if (!inserted) List.push(newNode);
            ListSet.insert(j);
        }
        while (List.size() > L) List.pop();
        priority_queue<tuple<float, size_t>> copy = List;
        diff = {};
        while (copy.size() != 0) {
            tuple<float, size_t> next = copy.top();
            copy.pop();
            bool exists = false;
            for (size_t k : Visited) {
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

void RobustPrune(Graph& graph, size_t point, vector<size_t>& candidates, long threshold, int R) {
//    cout << "In RobustPrune, point " << point << endl;
    set<size_t> neighbors = graph.mappings[point];
    for (size_t i : neighbors) {
        candidates.push_back(i);
    }
    for (size_t j = 0; j < candidates.size(); j++) {
        if (candidates[j] == point) {
            candidates[j] = candidates[candidates.size()-1];
            candidates.pop_back();
            break;
        }
    }
    graph.mappings[point] = {};
    while (candidates.size() != 0) {
        // find p* <- closest neighbor to p
        size_t bestCandidate = *candidates.begin();
        for (size_t j : candidates) {
            if (graph.findDistance(j, graph.nodes[point]) < graph.findDistance(bestCandidate, graph.nodes[point])) {
                bestCandidate = j;
            }
        }
        for (size_t j = 0; j < candidates.size(); j++) {
            if (candidates[j] == bestCandidate) {
                candidates[j] = candidates[candidates.size()-1];
                candidates.pop_back();
                break;
            }
        }
        // add best candidate back to p's neighborhood
        set<size_t> edges = graph.mappings[point];
        edges.insert(bestCandidate);
        graph.mappings[point] = edges;
        // neighborhood is full
        if (graph.mappings[point].size() == R) {
            break;
        }
        vector<size_t> copy;
        for (size_t k : candidates) {
            if (graph.findDistance(point, graph.nodes[k]) < threshold * graph.findDistance(bestCandidate, graph.nodes[k])) {
                copy.push_back(k);
            }
        }
        candidates = copy;
    }
}

size_t findStart(Config* config, const Graph& g) {
    float* center = new float[g.DIMENSION];
    for (size_t j = 0; j < g.num_nodes; j++) {
        for (size_t k = 0; k < g.DIMENSION; k++) {
            center[k] += g.nodes[j][k];
        }
    }
    for (size_t i = 0; i < g.DIMENSION; i++) {
        center[i] /= g.num_nodes;
    }
    size_t closest = 0;
    float closest_dist = MAXFLOAT;
    for (size_t m = 0; m < g.num_nodes; m++) {
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
    cout << "Start of Vamana" << endl;
    cout << "Randomizing edges" << endl;
    graph.randomize(R);
    cout << "Randomized edges" << endl;
    cout << "Random graph: " << endl;
    size_t s = findStart(config, graph);
    cout << "The centroid is #" << s << endl;
    for (int i = 0; i < 2; i++) {
        long actual_alpha = (i == 0) ? 1 : alpha;
        vector<size_t> sigma;
        for (size_t i = 0; i < config->num_nodes; i++) {
            sigma.push_back(i);
        }
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        shuffle(sigma.begin(), sigma.end(), default_random_engine(seed));
        size_t count = 0;
        for (size_t i : sigma) {
            if (count % 1000 == 0) cout << "Num of node processed: " << count << endl;
            count++;
            vector<size_t> result = GreedySearch(graph, s, graph.nodes[i], L);
            RobustPrune(graph, i, result, actual_alpha, R);
            set<size_t> neighbors = graph.mappings[i];
            for (size_t j : neighbors) {
                set<size_t> unionV = graph.mappings[j]; 
                vector<size_t> unionVec;
                for (size_t i : unionV) {
                    unionVec.push_back(i);
                }
                unionV.insert(i);
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

