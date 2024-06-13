#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>
#include "hnsw.h"

using namespace std;

// File name of fvecs file, including its path and excluding its extension
const string filename = "./exports/glove/train";
const int num_nodes = 1183514;
const int dim = 25;
const int hopkins_sample_size = 1000;
const int cluster_k = 400;
const int cluster_iterations = 20;

const bool COMPARE_DATASETS = false;
const string other_dataset = "./exports/glove/test";
const int other_num_nodes = 1000;

void load_fvecs(const string& file, const string& type, float** nodes, int num, int dim) {
    ifstream f(file, ios::binary | ios::in);
    if (!f) {
        cout << "File " << file << " not found!" << endl;
        exit(-1);
    }
    cout << "Loading " << num << " " << type << " from file " << file << endl;

    // Read dimension
    int read_dim;
    f.read(reinterpret_cast<char*>(&read_dim), 4);
    if (dim != read_dim) {
        cout << "Mismatch between expected and actual dimension: " << dim << " != " << read_dim << endl;
        exit(-1);
    }

    // Check size
    f.seekg(0, ios::end);
    if (num > f.tellg() / (dim * 4 + 4)) {
        cout << "Requested number of " << type << " is greater than number in file: "
            << num << " > " << f.tellg() / (dim * 4 + 4) << endl;
        exit(-1);
    }
    if (num != f.tellg() / (dim * 4 + 4)) {
        cout << "Warning: requested number of " << type << " is different from number in file: "
            << num << " != " << f.tellg() / (dim * 4 + 4) << endl;
    }

    f.seekg(0, ios::beg);
    for (int i = 0; i < num; i++) {
        // Skip dimension size
        f.seekg(4, ios::cur);

        // Read point
        nodes[i] = new float[dim];
        f.read(reinterpret_cast<char*>(nodes[i]), dim * 4);
    }
    f.close();
}

int count_same_nodes(float** nodes1, int size1, float** nodes2, int size2) {
    int count = 0;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            bool is_same = true;
            for (int d = 0; d < dim; d++) {
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

// Given some nodes, k, num, and dim, cluster the nodes and store its properties
// inside cluster_sizes and wcss (within-cluster sum of squares)
void k_means_cluster(float* cluster_sizes, float& wcss, float** nodes, int k, int num, int dim) {
    // Stop if k is invalid
    if (k < 1) {
        return;
    }

    // Generate random node indices for initial centroids
    mt19937 centroid_gen(std::chrono::system_clock::now().time_since_epoch().count());
    int* range = new int[num];
    for (int i = 0; i < num; i++) {
        range[i] = i;
    }
    std::shuffle(range, range + num, centroid_gen);

    // Initialize clusters
    float** centroids = new float*[k];
    float** dimension_totals = new float*[k];
    float* sum_of_squares = new float[k];
    for (int i = 0; i < k; i++) {
        centroids[i] = new float[dim];
        dimension_totals[i] = new float[dim];
        sum_of_squares[i] = 0;

        for (int d = 0; d < dim; d++) {
            centroids[i][d] = nodes[range[i]][d];
            dimension_totals[i][d] = nodes[range[i]][d];
        }
    }
    delete[] range;
    int* assignments = new int[num];

    for (int i = 0; i < cluster_iterations; i++) {
        // Update running statistics
        for (int m = 0; m < k; m++) {
            for (int d = 0; d < dim; d++) {
                dimension_totals[m][d] = 0;
            }
            cluster_sizes[m] = 0;
        }
        // Assign each node to the nearest cluster
        for (int n = 0; n < num; n++) {
            int min_distance = -1;
            int min_index = -1;
            for (int m = 0; m < k; m++) {
                float distance = calculate_l2_sq(nodes[n], centroids[m], dim, -1);
                if (min_distance == -1 || distance < min_distance) {
                    min_distance = distance;
                    min_index = m;
                }
            }
            assignments[n] = min_index;
            // Update new center of cluster
            for (int d = 0; d < dim; d++) {
                dimension_totals[min_index][d] += nodes[n][d];
            }
            cluster_sizes[min_index] += 1;
        }
        // Move centroids to center of cluster
        for (int m = 0; m < k; m++) {
            for (int d = 0; d < dim; d++) {
                centroids[m][d] = dimension_totals[m][d] / cluster_sizes[m];
            }
        }
    }
    // Calculate Within-Cluster Sum of Squares
    float total_error = 0;
    for (int i = 0; i < num; i++) {
        int cluster_index = assignments[i];
        sum_of_squares[cluster_index] +=  calculate_l2_sq(nodes[i], centroids[cluster_index], dim, -1);
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
float calculate_hopkins(float** nodes, int sample_size, int num, int dim, float* min, float* max) {
    // Obtain a random sample of nodes
    mt19937 sample_gen(std::chrono::system_clock::now().time_since_epoch().count());
    float** sample = new float*[sample_size];
    std::sample(nodes, nodes + num, sample, sample_size, sample_gen);

    // Generate uniform data using system clock time as generation seed
    mt19937 uniform_gen(std::chrono::system_clock::now().time_since_epoch().count());
    float** uniform = new float*[sample_size];
    for (int i = 0; i < sample_size; i++) {
        uniform[i] = new float[dim];
        for (int j = 0; j < dim; j++) {
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
            float distance = calculate_l2_sq(sample[i], sample[j], dim, -1);
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
            float distance = calculate_l2_sq(uniform[i], sample[j], dim, -1);
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

void calculate_stats(const string& name, float** nodes, int num, int dim, bool displayStats, bool exportStats, bool displayAggrStats) {
    // Calculate mean, median, and std of each dimension as well as min and max
    float* mean = new float[dim];
    float* median = new float[dim];
    float* std = new float[dim];

    float* min = new float[dim];
    float* max = new float[dim];

    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < num; j++) {
            sum += nodes[j][i];
        }
        mean[i] = sum / num;
    }

    for (int i = 0; i < dim; i++) {
        float* values = new float[num];
        for (int j = 0; j < num; j++) {
            values[j] = nodes[j][i];
        }
        sort(values, values + num);
        median[i] = values[num / 2];
        delete[] values;
    }

    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < num; j++) {
            sum += (nodes[j][i] - mean[i]) * (nodes[j][i] - mean[i]);
        }
        std[i] = sqrt(sum / num);
    }

    for (int i = 0; i < dim; i++) {
        float min_val = nodes[0][i];
        float max_val = nodes[0][i];
        for (int j = 1; j < num; j++) {
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

    float hopkins = calculate_hopkins(nodes, hopkins_sample_size, num, dim, min, max);

    float* cluster_sizes = new float[cluster_k];
    float wcss = 0;
    k_means_cluster(cluster_sizes, wcss, nodes, cluster_k, num, dim);

    int num_same = 0;
    if (COMPARE_DATASETS) {
        float** other_nodes = new float*[num_nodes];
        load_fvecs(other_dataset + ".fvecs", "base", other_nodes, other_num_nodes, dim);
        num_same = count_same_nodes(nodes, num_nodes, other_nodes, other_num_nodes);
    }
    
    if (displayStats) {
        cout << endl << "Mean: ";
        for (int i = 0; i < dim; i++) {
            cout << mean[i] << " ";
        }
        cout << endl;

        cout << endl << "Median: ";
        for (int i = 0; i < dim; i++) {
            cout << median[i] << " ";
        }
        cout << endl;

        cout << endl << "Std: ";
        for (int i = 0; i < dim; i++) {
            cout << std[i] << " ";
        }
        cout << endl;

        cout << endl << "Min: ";
        for (int i = 0; i < dim; i++) {
            cout << min[i] << " ";
        }
        cout << endl;

        cout << endl << "Max: ";
        for (int i = 0; i < dim; i++) {
            cout << max[i] << " ";
        }
        cout << endl;
    }

    if (exportStats) {
        ofstream f("runs/dataset_metrics.txt");
        f << num_nodes << " " << dim << endl;

        for (int i = 0; i < dim; i++) {
            f << mean[i] << " ";
        }
        f << endl;

        for (int i = 0; i < dim; i++) {
            f << median[i] << " ";
        }
        f << endl;

        for (int i = 0; i < dim; i++) {
            f << std[i] << " ";
        }
        f << endl;

        for (int i = 0; i < dim; i++) {
            f << min[i] << " ";
        }
        f << endl;

        for (int i = 0; i < dim; i++) {
            f << max[i] << " ";
        }
        f << endl;

        for (int i = 0; i < cluster_k; i++) {
            f << cluster_sizes[i] << " ";
        }

        f << endl << cluster_k << " " << cluster_iterations << " " << wcss;
        f << endl << hopkins << endl;

        if (COMPARE_DATASETS) {
            f << endl << num_same << endl;
        }

        f.close();
    }

    if (displayAggrStats) {
        // Sort mean, median, std, min, max, and cluster sizes, and display the top 10 and bottom 10 values
        sort(mean, mean + dim);
        sort(median, median + dim);
        sort(std, std + dim);
        sort(min, min + dim);
        sort(max, max + dim);
        sort(cluster_sizes, cluster_sizes + cluster_k);

        cout << endl << "Top 10 mean: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
            cout << mean[i] << " ";
        }

        cout << endl << "Bottom 10 mean: ";
        for (int i = 0; i < 10; i++) {
            cout << mean[i] << " ";
        }

        cout << endl << "Top 10 median: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
            cout << median[i] << " ";
        }

        cout << endl << "Bottom 10 median: ";
        for (int i = 0; i < 10; i++) {
            cout << median[i] << " ";
        }

        cout << endl << "Top 10 std: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
            cout << std[i] << " ";
        }

        cout << endl << "Bottom 10 std: ";
        for (int i = 0; i < 10; i++) {
            cout << std[i] << " ";
        }

        cout << endl << "Top 10 min: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
            cout << min[i] << " ";
        }

        cout << endl << "Bottom 10 min: ";
        for (int i = 0; i < 10; i++) {
            cout << min[i] << " ";
        }

        cout << endl << "Top 10 max: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
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
        for (int m = cluster_k - 1; m > cluster_k - 10; m--) {
            cout << cluster_sizes[m] << " ";
        }
        cout << endl << "Bottom 10 cluster sizes: ";
        for (int m = 0; m < 10; m++) {
            cout << cluster_sizes[m] << " ";
        }
        if (COMPARE_DATASETS) {
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

    float** nodes = new float*[num_nodes];
    load_fvecs(filename + ".fvecs", "base", nodes, num_nodes, dim);

    // Calculate stats
    cout << endl << "Base nodes:";
    calculate_stats(filename, nodes, num_nodes, dim, displayStats, exportStats, displayAggrStats);

    // Delete objects
    for (int i = 0; i < num_nodes; i++) {
        delete[] nodes[i];
    }
    delete[] nodes;

    return 0;
}
