#include <iostream>
#include <chrono>
#include "../HNSW/hnsw.h"

using namespace std;

const bool LOAD_FROM_FILE = false;
const string LOAD_DIR = "exports/";
const string LOAD_NAME = "random_graph";
const int LOAD_INDEX = 0;

const bool RUN_SEARCH = true;

/** 
 * This class is used to run a single instance of the HNSW algorithm.
*/
int main() {
    auto begin_time = chrono::high_resolution_clock::now();
    time_t now = time(NULL);
    cout << "HNSW run started at " << ctime(&now);

    Config* config = new Config();

    // Sanity checks
    if(!sanity_checks(config))
        return 1;

    // Get num_nodes amount of graph nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    cout << "Beginning HNSW construction" << endl;

    HNSW* hnsw = init_hnsw(config, nodes);
    if (LOAD_FROM_FILE) {
        // Check file and parameters
        const string graph_file_name = LOAD_DIR + LOAD_NAME + "_graph_" + to_string(LOAD_INDEX) + ".bin";
        const string info_file_name = LOAD_DIR + LOAD_NAME + "_info_" + to_string(LOAD_INDEX) + ".txt";
        ifstream graph_file(graph_file_name);
        ifstream info_file(info_file_name);
        cout << "Loading saved graph from " << graph_file_name << endl;

        if (!graph_file) {
            cout << "File " << graph_file_name << " not found!" << endl;
            return 1;
        }
        if (!info_file) {
            cout << "File " << info_file_name << " not found!" << endl;
            return 1;
        }

        int opt_con, max_con, max_con_0, ef_con;
        int num_nodes;
        int num_layers;
        info_file >> opt_con >> max_con >> max_con_0 >> ef_con;
        info_file >> num_nodes;
        info_file >> num_layers;

        // Check if number of nodes match
        if (num_nodes != config->num_nodes) {
            cout << "Mismatch between loaded and expected number of nodes" << endl;
            return 1;
        }

        // Check if construction parameters match
        if (opt_con != config->optimal_connections || max_con != config->max_connections ||
            max_con_0 != config->max_connections_0 || ef_con != config->ef_construction) {
            cout << "Mismatch between loaded and expected construction parameters" << endl;
            return 1;
        }

        hnsw->layers = num_layers;
        load_hnsw_graph(hnsw, graph_file, nodes, num_nodes, num_layers);
    } else {
        // Insert nodes into HNSW
        insert_nodes(config, hnsw);
    }

    // Print HNSW graph
    print_hnsw(config, hnsw);
    for (int i = 0; i < 11; i++) {
        vector<int> layer = get_layer(config, hnsw, i);
        cout << "Layer " << i << ", Size: " << layer.size() << endl;
    }

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

    // Export graph to file
    export_graph(config, hnsw, nodes);

    // Delete nodes
    for (int i = 0; i < config->num_nodes; i++)
        delete nodes[i];
    delete[] nodes;

    // Delete hnsw and config
    delete hnsw;
    delete config;

    now = time(NULL);
    cout << "HNSW run ended at " << ctime(&now);

    auto end_time = chrono::high_resolution_clock::now();
    cout << "Total time taken: " << chrono::duration_cast<chrono::milliseconds>(end_time - begin_time).count() << " ms" << endl;

    return 0;
}
