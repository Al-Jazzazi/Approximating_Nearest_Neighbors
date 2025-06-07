#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>

#include "../include/hnsw.h"

using namespace std;

int count_same_nodes(Config* config, float** nodes1, int size1, float** nodes2, int size2) {
    int count = 0;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            bool is_same = true;
            for (int d = 0; d < config->dimensions; d++) {
                if (nodes1[i][d] != nodes2[j][d]) {
                    is_same = false;
                    break;
                }
            }
            if (is_same) {
                count += 1;
                break;
            }
        }
    }
    return count;
}

/* Given some nodes, k, num, and dim, cluster the nodes and store its properties
 * inside cluster_sizes and wcss (within-cluster sum of squares)
 */
void k_means_cluster(Config* config, float* cluster_sizes, float& wcss, float** nodes, int k) {
    // Stop if k is invalid
    if (k < 1) {
        return;
    }

    // Generate random node indices for initial centroids
    mt19937 centroid_gen(std::chrono::system_clock::now().time_since_epoch().count());
    int* range = new int[config->num_nodes];
    for (int i = 0; i < config->num_nodes; i++) {
        range[i] = i;
    }
    std::shuffle(range, range + config->num_nodes, centroid_gen);

    // Initialize clusters
    float** centroids = new float*[k];
    float** dimension_totals = new float*[k];
    float* sum_of_squares = new float[k];
    for (int i = 0; i < k; i++) {
        centroids[i] = new float[config->dimensions];
        dimension_totals[i] = new float[config->dimensions];
        sum_of_squares[i] = 0;

        for (int d = 0; d < config->dimensions; d++) {
            centroids[i][d] = nodes[range[i]][d];
            dimension_totals[i][d] = nodes[range[i]][d];
        }
    }
    delete[] range;
    int* assignments = new int[config->num_nodes];

    for (int i = 0; i < config->cluster_iterations; i++) {
        // Update running statistics
        for (int m = 0; m < k; m++) {
            for (int d = 0; d < config->dimensions; d++) {
                dimension_totals[m][d] = 0;
            }
            cluster_sizes[m] = 0;
        }
        // Assign each node to the nearest cluster
        for (int n = 0; n < config->num_nodes; n++) {
            int min_distance = -1;
            int min_index = -1;
            for (int m = 0; m < k; m++) {
                float distance = calculate_l2_sq(nodes[n], centroids[m], config->dimensions);
                if (min_distance == -1 || distance < min_distance) {
                    min_distance = distance;
                    min_index = m;
                }
            }
            assignments[n] = min_index;
            // Update new center of cluster
            for (int d = 0; d < config->dimensions; d++) {
                dimension_totals[min_index][d] += nodes[n][d];
            }
            cluster_sizes[min_index] += 1;
        }
        // Move centroids to center of cluster
        for (int m = 0; m < k; m++) {
            for (int d = 0; d < config->dimensions; d++) {
                centroids[m][d] = dimension_totals[m][d] / cluster_sizes[m];
            }
        }
    }
    // Calculate Within-Cluster Sum of Squares
    float total_error = 0;
    for (int i = 0; i < config->num_nodes; i++) {
        int cluster_index = assignments[i];
        sum_of_squares[cluster_index] +=  calculate_l2_sq(nodes[i], centroids[cluster_index], config->dimensions);
    }
    for (int i = 0; i < k; i++) {
        if (cluster_sizes[i] > 0) {
            total_error += sum_of_squares[i] / cluster_sizes[i];
        }
    }
    wcss = total_error / k;

    // Clean up
    for (int m = 0; m < k; m++) {
        delete[] centroids[m];
        delete[] dimension_totals[m];
    }
    delete[] centroids;
    delete[] dimension_totals;
    delete[] sum_of_squares;
}

/**
 * Calculate Hopkins Statistic for the given nodes, where 0.5 indicates a
 * perfectly uniform dataset and 1.0 indicates a perfectly clustered dataset
*/
float calculate_hopkins(Config* config, float** nodes, int sample_size, float* min, float* max) {
    // Obtain a random sample of nodes
    mt19937 sample_gen(std::chrono::system_clock::now().time_since_epoch().count());
    float** sample = new float*[sample_size];
    std::sample(nodes, nodes + config->num_nodes, sample, sample_size, sample_gen);

    // Generate uniform data using system clock time as generation seed
    mt19937 uniform_gen(std::chrono::system_clock::now().time_since_epoch().count());
    float** uniform = new float*[sample_size];
    for (int i = 0; i < sample_size; i++) {
        uniform[i] = new float[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            uniform_real_distribution<float> distribution(min[j], max[j]);
            uniform[i][j] = distribution(uniform_gen);
        }
    }

    // Get distance from each real point to its real nearest neighbor
    float real_distance_sum = 0;
    float min_distance;
    for (int i = 0; i < sample_size; i++) {
        min_distance = -1;
        for (int j = 0; j < sample_size; j++) {
            float distance = calculate_l2_sq(sample[i], sample[j], config->dimensions);
            if ((min_distance == -1 || distance < min_distance) && i != j) {
                min_distance = distance;
            }
        }  
        real_distance_sum += min_distance;
    }

    // Get distance from each artificial point to its real nearest neighbor
    float artificial_distance_sum = 0;
    for (int i = 0; i < sample_size; i++) {
        min_distance = -1;
        for (int j = 0; j < sample_size; j++) {
            float distance = calculate_l2_sq(uniform[i], sample[j], config->dimensions);
            if ((min_distance == -1 || distance < min_distance)) {
                min_distance = distance;
            }
        }  
        artificial_distance_sum += min_distance;
    }

    // Clean up
    for (int i = 0; i < sample_size; i++) {
        delete uniform[i];
    }
    delete[] uniform;

    return artificial_distance_sum / (artificial_distance_sum + real_distance_sum);
}

void calculate_stats(Config* config, const string& name, float** nodes, bool displayStats, bool exportStats, bool displayAggrStats) {
    // Calculate mean, median, and std of each dimension as well as min and max
    float* mean = new float[config->dimensions];
    float* median = new float[config->dimensions];
    float* std = new float[config->dimensions];

    float* min = new float[config->dimensions];
    float* max = new float[config->dimensions];

    for (int i = 0; i < config->dimensions; i++) {
        float sum = 0;
        for (int j = 0; j < config->num_nodes; j++) {
            sum += nodes[j][i];
        }
        mean[i] = sum / config->num_nodes;
    }

    for (int i = 0; i < config->dimensions; i++) {
        float* values = new float[config->num_nodes];
        for (int j = 0; j < config->num_nodes; j++) {
            values[j] = nodes[j][i];
        }
        sort(values, values + config->num_nodes);
        median[i] = values[config->num_nodes / 2];
        delete[] values;
    }

    for (int i = 0; i < config->dimensions; i++) {
        float sum = 0;
        for (int j = 0; j < config->num_nodes; j++) {
            sum += (nodes[j][i] - mean[i]) * (nodes[j][i] - mean[i]);
        }
        std[i] = sqrt(sum / config->num_nodes);
    }

    for (int i = 0; i < config->dimensions; i++) {
        float min_val = nodes[0][i];
        float max_val = nodes[0][i];
        for (int j = 1; j < config->num_nodes; j++) {
            if (nodes[j][i] < min_val) {
                min_val = nodes[j][i];
            }
            if (nodes[j][i] > max_val) {
                max_val = nodes[j][i];
            }
        }
        min[i] = min_val;
        max[i] = max_val;
    }

    float hopkins = calculate_hopkins(config, nodes, config->hopkins_sample_size, min, max);

    float* cluster_sizes = new float[config->cluster_k];
    for (int i = 0; i < config->cluster_k; i++) {
        cluster_sizes[i] = 0;
    }
    float wcss = 0;
    k_means_cluster(config, cluster_sizes, wcss, nodes, config->cluster_k);

    int num_same = 0;
    if (config->compare_datasets) {
        float** comparison_nodes = new float*[config->comparison_num_nodes];
        load_fvecs(config->metrics_dataset2_prefix + ".fvecs", comparison_nodes, config->comparison_num_nodes, config->dimensions);
        num_same = count_same_nodes(config, nodes, config->num_nodes, comparison_nodes, config->comparison_num_nodes);
    }
    
    if (displayStats) {
        cout << endl << "Mean: ";
        for (int i = 0; i < config->dimensions; i++) {
            cout << mean[i] << " ";
        }
        cout << endl;

        cout << endl << "Median: ";
        for (int i = 0; i < config->dimensions; i++) {
            cout << median[i] << " ";
        }
        cout << endl;

        cout << endl << "Std: ";
        for (int i = 0; i < config->dimensions; i++) {
            cout << std[i] << " ";
        }
        cout << endl;

        cout << endl << "Min: ";
        for (int i = 0; i < config->dimensions; i++) {
            cout << min[i] << " ";
        }
        cout << endl;

        cout << endl << "Max: ";
        for (int i = 0; i < config->dimensions; i++) {
            cout << max[i] << " ";
        }
        cout << endl;
    }

    if (exportStats) {
        ofstream f(config->metrics_file);
        f << config->num_nodes << " " << config->dimensions << endl;

        for (int i = 0; i < config->dimensions; i++) {
            f << mean[i] << " ";
        }
        f << endl;

        for (int i = 0; i < config->dimensions; i++) {
            f << median[i] << " ";
        }
        f << endl;

        for (int i = 0; i < config->dimensions; i++) {
            f << std[i] << " ";
        }
        f << endl;

        for (int i = 0; i < config->dimensions; i++) {
            f << min[i] << " ";
        }
        f << endl;

        for (int i = 0; i < config->dimensions; i++) {
            f << max[i] << " ";
        }
        f << endl;

        for (int i = 0; i < config->cluster_k; i++) {
            f << cluster_sizes[i] << " ";
        }

        f << endl << config->cluster_k << " " << config->cluster_iterations << " " << wcss;
        f << endl << hopkins << endl;

        if (config->compare_datasets) {
            f << endl << num_same << endl;
        }

        f.close();
    }

    if (displayAggrStats) {
        // Sort mean, median, std, min, max, and cluster sizes, and display the top 10 and bottom 10 values
        sort(mean, mean + config->dimensions);
        sort(median, median + config->dimensions);
        sort(std, std + config->dimensions);
        sort(min, min + config->dimensions);
        sort(max, max + config->dimensions);
        sort(cluster_sizes, cluster_sizes + config->cluster_k);

        cout << endl << "Top 10 mean: ";
        for (int i = config->dimensions - 1; i >= config->dimensions - 10; i--) {
            cout << mean[i] << " ";
        }

        cout << endl << "Bottom 10 mean: ";
        for (int i = 0; i < 10; i++) {
            cout << mean[i] << " ";
        }

        cout << endl << "Top 10 median: ";
        for (int i = config->dimensions - 1; i >= config->dimensions - 10; i--) {
            cout << median[i] << " ";
        }

        cout << endl << "Bottom 10 median: ";
        for (int i = 0; i < 10; i++) {
            cout << median[i] << " ";
        }

        cout << endl << "Top 10 std: ";
        for (int i = config->dimensions - 1; i >= config->dimensions - 10; i--) {
            cout << std[i] << " ";
        }

        cout << endl << "Bottom 10 std: ";
        for (int i = 0; i < 10; i++) {
            cout << std[i] << " ";
        }

        cout << endl << "Top 10 min: ";
        for (int i = config->dimensions - 1; i >= config->dimensions - 10; i--) {
            cout << min[i] << " ";
        }

        cout << endl << "Bottom 10 min: ";
        for (int i = 0; i < 10; i++) {
            cout << min[i] << " ";
        }

        cout << endl << "Top 10 max: ";
        for (int i = config->dimensions - 1; i >= config->dimensions - 10; i--) {
            cout << max[i] << " ";
        }

        cout << endl << "Bottom 10 max: ";
        for (int i = 0; i < 10; i++) {
            cout << max[i] << " ";
        }

        cout << endl << "Hopkins Statistic: " << hopkins;

        // Print output
        cout << endl << "Within-Cluster Sum of Squares: " << wcss;
        cout << endl << "Top 10 cluster sizes: ";
        for (int m = config->cluster_k - 1; m > config->cluster_k - 10; m--) {
            cout << cluster_sizes[m] << " ";
        }
        cout << endl << "Bottom 10 cluster sizes: ";
        for (int m = 0; m < 10; m++) {
            cout << cluster_sizes[m] << " ";
        }
        if (config->compare_datasets) {
            cout << endl << "# of Common Nodes: " << num_same;
        }

        cout << endl;
    }

    delete[] mean;
    delete[] median;
    delete[] std;
    delete[] min;
    delete[] max;
}

int main() {
    bool displayStats = false;
    bool exportStats = true;
    bool displayAggrStats = true;
    Config* config = new Config();

    float** nodes = new float*[config->num_nodes];
    load_fvecs(config->metrics_dataset1_prefix + ".fvecs", nodes, config->num_nodes, config->dimensions);

    // Calculate stats
    cout << endl << "Base nodes:";
    calculate_stats(config, config->metrics_dataset1_prefix, nodes, displayStats, exportStats, displayAggrStats);

    // Delete objects
    for (int i = 0; i < config->num_nodes; i++) {
        delete[] nodes[i];
    }
    delete[] nodes;
    delete config;

    return 0;
}
