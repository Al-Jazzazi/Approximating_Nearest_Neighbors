#ifndef VAMANA_H
#define VAMANA_H

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <queue>

#include "config.h"
#include "utils.h"




class Vamana {
    friend std::ostream& operator<<(std::ostream& os, const Vamana& rhs);
public:
    // Node* allNodes;
    float** nodes;
    std::vector<std::set<int>> mappings;
    int num_nodes;
    int DIMENSION;
    int start;

    // Statistics
    long long int distanceCalculationCount;
    long long int num_original_termination;
    long long int num_distance_termination;

    std::vector<int> cur_groundtruth;

    Vamana(Config* config);
    ~Vamana();
    void toFiles(Config* config, const std::string& graph_name);
    void fromFiles(Config* config, bool is_benchmarking = false);
    void randomize(int R);
    float findDistance(int i, float* query) ;
    float findDistance(int i, int j) ; 
    void query(Config* config, int start, std::vector<std::vector<int>>& allResults, float** queries);
    void queryBruteForce(Config* config, int start);
    void sanityCheck(Config* config, const std::vector<std::vector<int>>& allResults) const;
    void queryTest(int start);
    void reset_statistics();
    bool should_terminate(Config* config, std::priority_queue<std::pair<float, int>>& top_k, std::pair<float, int>& top_1, float close_squared, float far_squared,  int candidates_popped_per_q);
    void calculate_termination(Config *config);
    

   
};
void RunSearch();
void randomEdges(Vamana& graph, int R);
std::vector<int> GreedySearch(Vamana& graph, int start, float* query, int L);
void RobustPrune(Vamana& graph, int point, std::vector<int>& candidates, long threshold, int R);
Vamana VamanaIndexing(Config* config, long alpha, int R);
int findStart(Config* config,  Vamana& g);
void print_100_nodes( Vamana& g, Config* config);
void BeamSearch(Vamana& graph, Config* config,int start,  float* query, int L, std::vector<int>& closest);
void runQueries(Config* config, Vamana& graph, float** queries);
void print_100_mappings(Vamana& graph, Config* config);
#endif