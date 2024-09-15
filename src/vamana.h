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
    std::vector<std::set<size_t>> mappings;
    int num_nodes;
    int DIMENSION;

    Graph(Config* config);
    ~Graph();
    void to_files(Config* config, const std::string& graph_name);
    void from_files(Config* config, bool is_benchmarking = false);
    void randomize(int R);
    float findDistance(size_t i, float* query) const;
    float findDistance(size_t i, size_t j) const; 
    void setEdge(size_t i, std::set<size_t> edges);
    std::vector<std::vector<size_t>> query(Config* config, size_t start);
    void queryBruteForce(Config* config, size_t start);
    void sanityCheck(Config* config, const std::vector<std::vector<size_t>>& allResults) const;
    void queryTest(size_t start);
   
};

void randomEdges(Graph& graph, int R);
std::vector<size_t> GreedySearch(Graph& graph, size_t start, float* query, size_t L);
void RobustPrune(Graph& graph, size_t point, std::vector<size_t>& candidates, long threshold, int R);
Graph Vamana(Config* config, long alpha, int L, int R);
size_t findStart(Config* config, const Graph& g);
void print_100_nodes(const Graph& g, Config* config);

#endif