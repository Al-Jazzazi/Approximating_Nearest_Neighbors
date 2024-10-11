#ifndef VAMANA_H
#define VAMANA_H

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include "utils.h"
#include "../config.h"



class Graph {
    friend std::ostream& operator<<(std::ostream& os, const Graph& rhs);
public:
    // Node* allNodes;
    float** nodes;
    std::vector<std::set<int>> mappings;
    int num_nodes;
    int DIMENSION;
    int start;

    Graph(Config* config);
    ~Graph();
    void to_files(Config* config, const std::string& graph_name);
    void from_files(Config* config, bool is_benchmarking = false);
    void randomize(int R);
    float findDistance(int i, float* query) const;
    float findDistance(int i, int j) const; 
    void setEdge(int i, std::set<int> edges);
    void query(Config* config, int start, std::vector<std::vector<int>>& allResults, float** queries);
    void queryBruteForce(Config* config, int start);
    void sanityCheck(Config* config, const std::vector<std::vector<int>>& allResults) const;
    void queryTest(int start);
   
};

void randomEdges(Graph& graph, int R);
std::vector<int> GreedySearch(Graph& graph, int start, float* query, int L);
void RobustPrune(Graph& graph, int point, std::vector<int>& candidates, long threshold, int R);
Graph Vamana(Config* config, long alpha, int L, int R);
int findStart(Config* config, const Graph& g);
void print_100_nodes(const Graph& g, Config* config);
void BeamSearch(Graph& graph, Config* config,int start,  float* query, int L, std::vector<int> closest);
void get_actual_neighbors(Config* config, std::vector<std::vector<int>>& actual_neighbors, float** nodes, float** queries);
void runQueries(Config* config, Graph& graph, float** queries);
void print_100_mappings(const Graph& graph, Config* config);
#endif