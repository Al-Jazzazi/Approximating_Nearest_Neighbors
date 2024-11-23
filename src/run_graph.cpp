#include "graph.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <chrono>



using namespace std;
int main() {
    Config* config = new Config();
    Graph G(config);
    auto start = std::chrono::high_resolution_clock::now();
    G.load(config); 
    auto end = chrono::high_resolution_clock::now();;
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Load time: " << duration / 1000.0 << " seconds, ";

    float** queries = new float*[config->num_queries];
    load_queries(config, G.nodes, queries);

    G.runQueries(config, queries);
    G.print_100_mappings(config);
    // G.print_k_nodes(config);
    // G.print_k_neigbours(config, 10);

     for (int i = 0; i < config->num_queries; ++i)
        delete[] queries[i];
    delete[] queries;

    delete config;

    return 0;
}
