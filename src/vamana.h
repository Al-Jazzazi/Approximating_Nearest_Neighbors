#ifndef VAMANA_H
#define VAMANA_H

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include "utils.h"
#include "../config.h"

class DataNode {
    friend std::ostream& operator<<(std::ostream& os, const DataNode& rhs);
    friend bool operator==(const DataNode& lhs, const DataNode& rhs);
public:
    DataNode();
    DataNode(double* coord);
//    void sumArraysAVX(float* array1, float* array2, float* result, int size) const;
    long long int findDistanceAVX(const DataNode& other) const;
    double findDistance(const DataNode& other) const;
    bool compare(double* coord) const;
    void addCoord(double* coord) const;
    void setWord(const std::string& theWord);
private:
    size_t dimension;
    double* coordinates;
    std::string word;
};

struct Node {
    DataNode val;
    std::set<size_t> outEdge;
};

class Graph {
public:
    Graph(int total);
    size_t findNode(const DataNode& val);
    void addNode(const DataNode& val, std::set<size_t>& neighbors, size_t pos);
    void randomize(int R);
    std::set<size_t> getNeighbors(const DataNode& i);
    void clearNeighbors(size_t i);
    double findDistance(size_t i, const DataNode& query) const;
    Node getNode(size_t i) const;
    std::set<size_t> getNodeNeighbor(size_t i) const;
    void setEdge(size_t i, std::set<size_t> edges);
    void display() const;
    std::vector<std::vector<size_t>> query(Config* config, size_t start);
    void queryBruteForce(Config* config, size_t start);
    void sanityCheck(Config* config, std::vector<std::vector<size_t>> allResults) const;
    void queryTest(size_t start);
private:
    Node* allNodes;
};

void constructGraph(std::vector<DataNode>& allNodes, Graph& graph);
void randomEdges(Graph& graph, int R);
template<typename T>
bool findInSet(const std::set<T>& set, T target);
template<typename Y>
std::set<Y> setDiff(const std::set<Y>& setOne, const std::set<Y>& setTwo);
std::vector<size_t> GreedySearch(Graph& graph, size_t start, const DataNode& query, size_t L);
void RobustPrune(Graph& graph, size_t point, std::vector<size_t>& candidates, long threshold, int R);
Graph Vamana(std::vector<DataNode>& allNodes, long alpha, int L, int R);
void getNodes(std::vector<DataNode>& allNodes, const std::string& fileName, size_t dimension);
void getNodesGlove(std::vector<DataNode>& allNodes, const std::string& fileName, size_t dimension);
size_t findStart(const Graph& g);

#endif