#ifndef VAMANA_H
#define VAMANA_H

#include <iostream>
#include <vector>
#include <set>
#include <string>

class DataNode {
    friend std::ostream& operator<<(std::ostream& os, const DataNode& rhs);
    friend bool operator==(const DataNode& lhs, const DataNode& rhs);
public:
    size_t dimension;
    double* coordinates;
    std::string word;

    DataNode();
    DataNode(double* coord);
    long long int findDistanceAVX(const DataNode& other) const;
    double findDistance(const DataNode& other) const;
    bool compare(double* coord) const;
    void addCoord(double* coord) const;
    void setWord(const std::string& theWord);
};

struct Node {
    DataNode val;
    std::set<size_t> outEdge;
};

class Graph {
    friend std::ostream& operator<<(std::ostream& os, const Graph& rhs);
public:
    Node* allNodes = nullptr;

    Graph();
    ~Graph();
    size_t findNode(const DataNode& val);
    void addNode(const DataNode& val, std::set<size_t>& neighbors, size_t pos);
    void randomize(int R);
    std::set<size_t> getNeighbors(const DataNode& i);
    void clearNeighbors(size_t i);
    double findDistance(size_t i, const DataNode& query) const;
    Node getNode(size_t i) const;
    std::set<size_t> getNodeNeighbor(size_t i) const;
    void setEdge(size_t i, std::set<size_t> edges);
    std::vector<std::vector<size_t>> query(size_t start);
    void queryBruteForce(size_t start);
    void sanityCheck(std::vector<std::vector<size_t>> allResults) const;
    void queryTest(size_t start);
};

void constructGraph(std::vector<DataNode>& allNodes, Graph& graph);
void randomEdges(Graph& graph, int R);
std::vector<size_t> GreedySearch(Graph& graph, size_t start, const DataNode& query, size_t L);
void RobustPrune(Graph& graph, size_t point, std::vector<size_t>& candidates, long threshold, int R);
Graph Vamana(std::vector<DataNode>& allNodes, long alpha, int L, int R);
void load_fvecs(std::vector<DataNode>& allNodes, const std::string& file);
size_t findStart(const Graph& g);
template<typename T>
bool findInSet(const std::set<T>& set, T target);
template<typename T>
std::set<T> setDiff(const std::set<T>& setOne, const std::set<T>& setTwo);

#endif