#include <immintrin.h>
#include <fstream>
#include <queue>
#include <cpuid.h>
#include <random>
#include <chrono>
#include <unordered_set>
#include "utils.h"

using namespace std;

// Calculates the squared Euclidean distance between points a and b
float calculate_l2_sq(float* a, float* b, int dimensions) {
    int parts = dimensions / 8;

    // Initialize result to 0
    __m256 result = _mm256_setzero_ps();

    // Process 8 floats at a time
    for (size_t i = 0; i < parts; ++i) {
        // Load vectors from memory into AVX registers
        __m256 vec_a = _mm256_loadu_ps(&a[i * 8]);
        __m256 vec_b = _mm256_loadu_ps(&b[i * 8]);

        // Compute differences and square
        __m256 diff = _mm256_sub_ps(vec_a, vec_b);
        __m256 diff_sq = _mm256_mul_ps(diff, diff);

        result = _mm256_add_ps(result, diff_sq);
    }

    // Process remaining floats
    float remainder = 0;
    for (size_t i = parts * 8; i < dimensions; ++i) {
        float diff = a[i] - b[i];
        remainder += diff * diff;
    }

    // Sum all floats in result
    float sum[8];
    _mm256_storeu_ps(sum, result);
    for (size_t i = 1; i < 8; ++i) {
        sum[0] += sum[i];
    }

    return sum[0] + remainder;
}

// Obtain the actual nearest neighbors either using groundtruth file or exact KNN search 
void get_actual_neighbors(Config* config, vector<vector<int>>& actual_neighbors, float** nodes, float** queries) {
    bool use_groundtruth = config->groundtruth_file != "";
    if (use_groundtruth && config->query_file == "") {
        cout << "Warning: Groundtruth file will not be used because queries were generated" << endl;
        use_groundtruth = false;
    }
    if (use_groundtruth) {
        // Load actual nearest neighbors
        load_ivecs(config->groundtruth_file, actual_neighbors, config->num_queries, config->num_return);
    } else {
        // Calcuate actual nearest neighbors
        auto start = chrono::high_resolution_clock::now();
        knn_search(config, actual_neighbors, nodes, queries);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Brute force time: " << duration / 1000.0 << " seconds" << endl;
    }
    if (config->benchmark_print_neighbors) {
        for (int i = 0; i < config->num_queries; ++i) {
            cout << "Neighbors in ideal case for query " << i << endl;
            for (size_t j = 0; j < actual_neighbors[i].size(); ++j) {
                float dist = calculate_l2_sq(queries[i], nodes[actual_neighbors[i][j]], config->dimensions);
                cout << actual_neighbors[i][j] << " (" << dist << ") ";
            }
            cout << endl;
        }
    }
}



// Finds the nearest neighbors from nodes to each query using an exact KNN search
void knn_search(Config* config, vector<vector<int>>& results, float** nodes, float** queries) {
    results.resize(config->num_queries);
    for (int i = 0; i < config->num_queries; ++i) {
        // Fill priority queue with nodes
        priority_queue<pair<float, int>> pq;
        for (int j = 0; j < config->num_nodes; ++j) {
            float dist = calculate_l2_sq(queries[i], nodes[j], config->dimensions);
            pq.emplace(dist, j);
            if (pq.size() > config->num_return) {
                pq.pop();
            }
        }
        // Place priority queue elements into results
        results[i].resize(config->num_return);
        size_t idx = pq.size();
        while (idx > 0) {
            --idx;
            results[i][idx] = pq.top().second;
            pq.pop();
        }
    }
}

// Loads num vectors with dim values from fvecs file
void load_fvecs(const string& file, float** vectors, int num, int dim, bool check_groundtruth) {
    // Open file
    ifstream f(file, ios::binary | ios::in);
    if (!f) {
        cout << "File " << file << " not found!" << endl;
        exit(-1);
    }
    cout << "Loading " << num << " vectors from file " << file << endl;

    // Verify dimension
    int read_dim;
    f.read(reinterpret_cast<char*>(&read_dim), 4);
    if (dim != read_dim) {
        cout << "Mismatch between expected and actual dimension: " << dim << " != " << read_dim << endl;
        exit(-1);
    }

    // Verify number of vectors
    f.seekg(0, ios::end);
    if (num > f.tellg() / (dim * 4 + 4)) {
        cout << "Requested number of vectors is greater than number in file: "
             << num << " > " << f.tellg() / (dim * 4 + 4) << endl;
        exit(-1);
    }
    if (num != f.tellg() / (dim * 4 + 4) && check_groundtruth) {
        cout << "You must load all " << f.tellg() / (dim * 4 + 4) << " nodes if you want to use a groundtruth file" << endl;
        exit(-1);
    }

    // Load data into vectors
    f.seekg(0, ios::beg);
    for (int i = 0; i < num; i++) {
        // Skip dimension size
        f.seekg(4, ios::cur);
        // Read point
        vectors[i] = new float[dim];
        f.read(reinterpret_cast<char*>(vectors[i]), dim * 4);
    }
    f.close();
}

// Saves num vectors with dim values into fvecs file
void save_fvecs(const string& file, float** vectors, int num, int dim) {
    ofstream f(file, ios::binary | ios::out);
    if (!f) {
        cout << "Unable to open file " << file << " for writing!" << endl;
        exit(-1);
    }
    cout << "Saving " << num << " vectors to file " << file << endl;

    for (int i = 0; i < num; i++) {
        f.write(reinterpret_cast<const char*>(&dim), 4);
        // Write the vector
        f.write(reinterpret_cast<const char*>(vectors[i]), dim * 4);
    }
    f.close();
}

// Loads num vectors with num_return values from ivecs file
void load_ivecs(const string& file, vector<vector<int>>& vectors, int num, int num_return) {
    // Open file
    ifstream f(file, ios::binary | ios::in);
    if (!f) {
        cout << "File " << file << " not found!" << endl;
        exit(-1);
    }
    cout << "Loading groundtruth from file " << file << endl;
    // Verify num_return
    int read_num_return;
    f.read(reinterpret_cast<char*>(&read_num_return), 4);
    if (num_return > read_num_return) {
        cout << "Requested num_return is greater than width in file: " << num_return << " > " << read_num_return << endl;
        exit(-1);
    }
    // Verify number of vectors
    f.seekg(0, ios::end);
    if (num > f.tellg() / (read_num_return * 4 + 4)) {
        cout << "Requested number of queries is greater than number in file: "
             << num << " > " << f.tellg() / (read_num_return * 4 + 4) << endl;
        exit(-1);
    }
    // Load data into vectors
    vectors.reserve(num);
    f.seekg(0, ios::beg);
    for (int i = 0; i < num; i++) {
        // Skip list size
        f.seekg(4, ios::cur);
        // Read point
        int values[num_return];
        f.read(reinterpret_cast<char*>(values), num_return * 4);
        vectors.push_back(vector<int>(values, values + num_return));
        // Skip remaining values
        f.seekg((read_num_return - num_return) * 4, ios::cur);
    }
    f.close();
}

// Save vectors to ivecs file
void save_ivecs(const string& file, vector<vector<int>>& vectors) {
    // Open file
    ofstream f(file, ios::binary | ios::out);
    if (!f) {
        cout << "Unable to open " << file << " for writing" << endl;
        exit(-1);
    }
    cout << "Saving groundtruth to file " << file << endl;
    // Write vectors into file
    int num_return = vectors[0].size();
    for (int i = 0; i < vectors.size(); ++i) {
        if (vectors[i].size() != num_return) {
            cout << "Inconsistent vector sizes detected!" << endl;
            exit(-1);
        }
        f.write(reinterpret_cast<const char*>(&num_return), 4);
        f.write(reinterpret_cast<const char*>(vectors[i].data()), num_return * 4);
    }
    f.close();
}

// Loads nodes from text file, fvecs file, or random generation
void load_nodes(Config* config, float** nodes) {
    if (config->load_file != "") {
        // Load nodes from fvecs file
        if (config->load_file.size() >= 6 && config->load_file.substr(config->load_file.size() - 6) == ".fvecs") {
            load_fvecs(config->load_file, nodes, config->num_nodes, config->dimensions, config->groundtruth_file != "");
            return;
        }
        // Load nodes from text file
        ifstream f(config->load_file, ios::in);
        if (!f) {
            cout << "File " << config->load_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << config->num_nodes << " nodes from file " << config->load_file << endl;
        for (int i = 0; i < config->num_nodes; i++) {
            nodes[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> nodes[i][j];
            }
        }
        f.close();
        return;
    }

    // Generate nodes
    cout << "Generating " << config->num_nodes << " random nodes" << endl;
    mt19937 gen(config->graph_seed);
    uniform_real_distribution<float> dis(config->gen_min, config->gen_max);
    for (int i = 0; i < config->num_nodes; i++) {
        nodes[i] = new float[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            nodes[i][j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
    }
}

// Loads queries from text file, fvecs file, or random generation
void load_queries(Config* config, float** nodes, float** queries) {
    mt19937 gen(config->query_seed);
    if (config->query_file != "") {
        // Load queries from fvecs file
        if (config->query_file.size() >= 6 && config->query_file.substr(config->query_file.size() - 6) == ".fvecs") {
            load_fvecs(config->query_file, queries, config->num_queries, config->dimensions);
            return;
        }

        // Load queries from text file
        ifstream f(config->query_file, ios::in);
        if (!f) {
            cout << "File " << config->query_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << config->num_queries << " queries from file " << config->query_file << endl;
        for (int i = 0; i < config->num_queries; i++) {
            queries[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> queries[i][j];
            }
        }
        f.close();
        return;
    }
    
    // Find the bounds of base nodes
    cout << "Generating queries based on file " << config->load_file << endl;
    float* lower_bound = new float[config->dimensions];
    float* upper_bound = new float[config->dimensions];
    copy(nodes[0], nodes[0] + config->dimensions, lower_bound);
    copy(nodes[0], nodes[0] + config->dimensions, upper_bound);
    for (int i = 1; i < config->num_nodes; i++) {
        for (int j = 0; j < config->dimensions; j++) {
            if (nodes[i][j] < lower_bound[j]) {
                lower_bound[j] = nodes[i][j];
            }
            if (nodes[i][j] > upper_bound[j]) {
                upper_bound[j] = nodes[i][j];
            }
        }
    }

    // Generate queries randomly based on bounds of graph_nodes
    uniform_real_distribution<float>* dis_array = new uniform_real_distribution<float>[config->dimensions];
    for (int i = 0; i < config->dimensions; i++) {
        dis_array[i] = uniform_real_distribution<float>(lower_bound[i], upper_bound[i]);
    }
    for (int i = 0; i < config->num_queries; i++) {
        queries[i] = new float[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            queries[i][j] = round(dis_array[j](gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
    }
    delete[] lower_bound;
    delete[] upper_bound;
    delete[] dis_array;
}

// Loads oracle file of query indices and distance calculations into results
void load_oracle(Config* config, vector<pair<int, int>>& results) {
    // Open file
    cout << "Loading oracle file: " << config->oracle_file;
    ifstream f(config->oracle_file);
    if (!f) {
        cout << config->oracle_file << " not found";
        exit(-1);
    }
    // Find and sort queries by distance calculations
    int calculation = 0;
    int index = 0;
    while (f >> calculation) {
        if (calculation == -1) {
            calculation = std::numeric_limits<int>::max();
        }
        results.push_back(make_pair(calculation, index));
        ++index;
    }
    f.close();
    std::sort(results.begin(), results.end());
}


string get_cpu_brand() {
    char CPUBrand[0x40];
    unsigned int CPUInfo[4] = {0,0,0,0};
    __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    unsigned int nExIds = CPUInfo[0];
    memset(CPUBrand, 0, sizeof(CPUBrand));

    for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
        __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
        if (i == 0x80000002)
            memcpy(CPUBrand, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            memcpy(CPUBrand + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            memcpy(CPUBrand + 32, CPUInfo, sizeof(CPUInfo));
    }
    string output(CPUBrand);
    return output;
}




void find_similar(Config* config,  const vector<vector<int>> actual_neighbors, const vector<vector<int>> neighbors,  float** nodes, float** queries, int& similar, float& total_ndcg) {
     for (int j = 0; j < config->num_queries; ++j) {
                // Find similar neighbors
                unordered_set<int> actual_set(actual_neighbors[j].begin(), actual_neighbors[j].end());
                unordered_set<int> intersection;
                float actual_gain = 0;
                float ideal_gain = 0;

                for (size_t k = 0; k < neighbors[j].size(); ++k) {
                    int n = neighbors[j][k];
                    float gain = 1 / log2(k + 2);
                    ideal_gain += gain;
                    if (actual_set.find(n) != actual_set.end()) {
                        intersection.insert(n);
                        actual_gain += gain;
                    }
                }
                similar += intersection.size();
                total_ndcg += actual_gain / ideal_gain;

                // Print out neighbors[i][j]
                if (config->benchmark_print_neighbors) {
                    cout << "Neighbors for query " << j << ": ";
                    for (size_t k = 0; k < neighbors[j].size(); ++k) {
                        auto n = neighbors[j][k];
                        cout << n;
                    }
                    cout << endl;
                }

                // Print missing neighbors between intersection and actual_neighbors
                if (config->benchmark_print_missing) {
                    cout << "Missing neighbors for query " << j << ": ";
                    if (intersection.size() == actual_neighbors[j].size()) {
                        cout << "None" << endl;
                        continue;
                    }
                    for (size_t k = 0; k < actual_neighbors[j].size(); ++k) {
                        if (intersection.find(actual_neighbors[j][k]) == intersection.end()) {
                            float dist = calculate_l2_sq(queries[j], nodes[actual_neighbors[j][k]], config->dimensions);
                            cout << actual_neighbors[j][k] << " (" << dist << ") ";
                        }
                    }
                    cout << endl;
                }
            }
}

