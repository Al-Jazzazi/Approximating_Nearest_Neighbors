#ifndef GRAPH_H
#define GRAPH_H
#include<vector> 
#include<set> 
#include "../config.h"
#include <queue>
#include<unordered_set>

class Graph {
public:
    // Node* allNodes;
    float** nodes;
    std::vector<std::set<unsigned>> mappings;
    int num_nodes;
    int DIMENSION;
    unsigned start;
    unsigned width;
    unsigned ep_;


    long long int distanceCalculationCount;
    long long int num_original_termination;
    long long int num_distance_termination;

    std::vector<int> cur_groundtruth;

    Graph(Config* config);
    ~Graph();
    void load(Config* config);

    float findDistance(int i, float* query) ;
    float findDistance(int i, int j) ; 
    void reset_statistics();

    bool should_terminate(Config* config, std::priority_queue<std::pair<float, int>>& top_k, std::pair<float, int>& top_1, float close_squared, float far_squared,  int candidates_popped_per_q);
    void calculate_termination(Config *config);
    void  runQueries(Config* config, float** queries);
    void query(Config* config, int start, std::vector<std::vector<int>>& allResults, float** queries);
    void print_100_mappings(Config* config);

};

void BeamSearch(Graph& graph, Config* config,int start,  float* query, int bw, std::vector<int>& closest);



#endif