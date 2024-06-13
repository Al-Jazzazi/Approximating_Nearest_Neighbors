#include <iostream>
#include <chrono>
#include "../HNSW/hnsw.h"

using namespace std;

const bool LOAD_FROM_FILE = false;
const string GRAPH_FILE = "exports/";
const string INFO_FILE = "exports/";
const bool RUN_SEARCH = true;

int main() {
    // Initialize time and config
    auto begin_time = chrono::high_resolution_clock::now();
    time_t now = time(NULL);
    cout << "GraSP run started at " << ctime(&now);
    Config* config = new Config();
    if(!sanity_checks(config))
        return 1;

    // Construct HNSW
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    cout << "Beginning HNSW construction" << endl;
    HNSW* hnsw = init_hnsw(config, nodes);
    if (LOAD_FROM_FILE) {
        load_hnsw_file(config, hnsw, nodes, GRAPH_FILE, INFO_FILE);
    } else {
        insert_nodes(config, hnsw);
    }

    // Optimize HNSW using GraSP
    

    // Run queries
    if (RUN_SEARCH) {
        // Generate num_queries amount of queries
        float** queries = new float*[config->num_queries];
        load_queries(config, nodes, queries);
        auto search_start = chrono::high_resolution_clock::now();
        cout << "Time passed: " << chrono::duration_cast<chrono::milliseconds>(search_start - begin_time).count() << " ms" << endl;
        cout << "Beginning search" << endl;

        // Run query search and print results
        run_query_search(config, hnsw, queries);

        auto search_end = chrono::high_resolution_clock::now();
        cout << "Time passed: " << chrono::duration_cast<chrono::milliseconds>(search_end - search_start).count() << " ms" << endl;

        // Delete queries
        for (int i = 0; i < config->num_queries; ++i)
            delete queries[i];
        delete[] queries;
    }

    // Clean up
    for (int i = 0; i < config->num_nodes; i++)
        delete nodes[i];
    delete[] nodes;
    delete hnsw;
    delete config;

    // Print time elapsed
    now = time(NULL);
    cout << "GraSP run ended at " << ctime(&now);
    auto end_time = chrono::high_resolution_clock::now();
    cout << "Total time taken: " << chrono::duration_cast<chrono::milliseconds>(end_time - begin_time).count() << " ms" << endl;

    return 0;
}
