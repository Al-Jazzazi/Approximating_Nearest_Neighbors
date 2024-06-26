#include <iostream>
#include <algorithm>
#include <chrono>
#include "hnsw.h"

using namespace std;

/**
 * This class is used to export multiple HNSW graphs to file.
*/
int main() {
    time_t now = time(NULL);
    cout << "HNSW save run started at " << ctime(&now);
    Config* config = new Config();

    // Get num_nodes amount of graph nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    cout << "Construction parameters: opt_con, max_con, max_con_0, ef_con" << endl;

    // Run HNSW with different ef_construction values
    for (int i = 0; i < config->num_graphs_saved; ++i) {
        config->optimal_connections = config->save_optimal_connections[i];
        config->max_connections = config->save_max_connections[i];
        config->max_connections_0 = config->save_max_connections_0[i];
        config->ef_construction = config->save_ef_constructions[i];
        layer0_dist_comps = 0;
        upper_dist_comps = 0;

        // Sanity checks
        if(!config->sanity_checks()) {
            cout << "Config error!" << endl;
            return 1;
        }

        auto start = chrono::high_resolution_clock::now();

        // Insert nodes into HNSW
        cout << "Inserting with construction parameters: "
            << config->optimal_connections << ", " << config->max_connections << ", "
            << config->max_connections_0 << ", " << config->ef_construction << endl; 
        HNSW* hnsw = init_hnsw(config, nodes);
        for (int i = 1; i < config->num_nodes; i++) {
            hnsw->insert(config, i);
        }

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Time taken: " << duration / 1000.0 << " seconds" << endl;
        cout << "Distance computations (layer 0): " << layer0_dist_comps << endl;
        cout << "Distance computations (top layers): " << upper_dist_comps << endl;

        // Export graph to file
        ofstream graph_file(config->save_file_prefix + "_graph_" + to_string(i) + ".bin");

        // Export edges
        for (int i = 0; i < config->num_nodes; ++i) {
            int layers = hnsw->mappings[i].size();

            // Write number of layers
            graph_file.write(reinterpret_cast<const char*>(&layers), sizeof(layers));

            // Write layers
            for (int j = 0; j < layers; ++j) {
                int num_neighbors = hnsw->mappings[i][j].size();

                // Write number of neighbors
                graph_file.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));

                // Write neighbors
                for (int k = 0; k < num_neighbors; ++k) {
                    auto n_pair = hnsw->mappings[i][j][k];
                    
                    // Write index and distance
                    graph_file.write(reinterpret_cast<const char*>(&n_pair.target), sizeof(n_pair.target));
                    graph_file.write(reinterpret_cast<const char*>(&n_pair.distance), sizeof(n_pair.distance));
                }
            }
        }

        // Save entry point
        graph_file.write(reinterpret_cast<const char*>(&hnsw->entry_point), sizeof(hnsw->entry_point));
        graph_file.close();

        // Export construction parameters
        ofstream info_file(config->save_file_prefix + "_info_" + to_string(i) + ".txt");
        info_file << config->optimal_connections << " " << config->max_connections << " "
            << config->max_connections_0 << " " << config->ef_construction << endl;
        info_file << config->num_nodes << endl;
        info_file << hnsw->num_layers << endl;
        info_file << layer0_dist_comps << endl;
        info_file << upper_dist_comps << endl;
        info_file << duration / 1000.0 << endl;

        cout << "Exported graph to " << config->save_file_prefix + "_graph_" + to_string(i) + ".bin" << endl;

        delete hnsw;
    }

    // Delete nodes
    for (int i = 0; i < config->num_nodes; i++)
        delete nodes[i];
    delete[] nodes;

    now = time(NULL);
    cout << "HNSW save run ended at " << ctime(&now);

    return 0;
}
