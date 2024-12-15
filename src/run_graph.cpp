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

    vector<vector<int>> actualResults;
    // get_actual_neighbors(config, actualResults, G.nodes, queries);
    G.calculate_termination(config);
    G.runQueries(config, queries);
    // G.print_100_mappings(config);
    // G.print_avg_neigbor(config);
    // G.print_k_nodes(config);
    // G.print_k_neigbours(config, 10);


    std::cout << "\nDistance Calc " << G.distanceCalculationCount/config->num_queries << std::endl; 
     for (int i = 0; i < config->num_queries; ++i)
        delete[] queries[i];
    delete[] queries;

    delete config;

    return 0;
}
