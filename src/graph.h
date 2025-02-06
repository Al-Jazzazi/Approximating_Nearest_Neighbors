#ifndef GRAPH_H
#define GRAPH_H
#include<vector> 
#include<set> 
#include "../config.h"
#include <queue>
#include<unordered_set>
/*
Graph files are used to run Efanna and NSG graphs (and hopefully any other graphs added later)
They're an abstraction from many of the functions that are used in HNSW where I removed
the segments of code related to multilayers graph, grasp pruning, and other test materials we scrapped 

The graph files contain: 
graph.h // graph.cpp 
run_graph.cpp to run the the graph through the variables set in config.h and outputting to terminal 
grraph_benchmark.cpp to run the graph through multiple changing variable and outputting to a .txt file
*/


class Graph {
public:
    float** nodes;
    std::vector<std::set<unsigned>> mappings;
    int num_nodes;
    int DIMENSION;
    unsigned start;
    unsigned width;


    long long int distanceCalculationCount;
    long long int num_original_termination;
    long long int num_distance_termination;
    long long int num_set_checks;
    long long int size_of_c;
    long long int num_insertion_to_c;
    long long int num_deletion_from_c;
    long long int size_of_visited;
    

    std::vector<int> cur_groundtruth;

    Graph(Config* config);
    ~Graph();
    void load(Config* config);


    float find_distance(int i, float* query) ;
    float find_distance(int i, int j) ; 
    void reset_statistics();

    bool should_terminate(Config* config, std::priority_queue<std::pair<float, int>>& top_k, std::pair<float, int>& top_1, float close_squared, float far_squared,  int candidates_popped_per_q);
    void calculate_termination(Config *config);
    void run_queries(Config* config, float** queries);
    void query(Config* config, int start, std::vector<std::vector<int>>& allResults, float** queries);
   
    //Debugging functions 
    void print_100_mappings(Config* config);
    void print_k_nodes( Config* config, int k = 100);
    void print_k_neigbours(Config* config, int k = 100);
    void print_avg_neigbor(Config* config);
};

void beam_search(Graph& graph, Config* config,int start,  float* query, int bw, std::vector<int>& closest);



#endif