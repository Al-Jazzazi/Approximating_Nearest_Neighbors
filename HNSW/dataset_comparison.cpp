#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>
#include "hnsw.h"

using namespace std;

// File name of fvecs file, including its path and excluding its extension
const string dataset1 = "./exports/gist/gist_base";
const string dataset2 = "./exports/gist/gist_learn";
const int size1 = 1000000;
const int size2 = 500000;
const int dim = 960;

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

int count_same_nodes(ofstream& file, float** nodes1, int size1, float** nodes2, int size2) {
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
        if (i % 10000 == 0) {
            file << "Iteration " << i << endl;
        }
    }
    return count;
}

int main() {
    ofstream f("exports/comparison.txt");
    float** nodes1 = new float*[size1];
    float** nodes2 = new float*[size2];
    load_fvecs(dataset1 + ".fvecs", "base", nodes1, size1, dim);
    load_fvecs(dataset2 + ".fvecs", "query", nodes2, size2, dim);
    
    int count = count_same_nodes(f, nodes1, size1, nodes2, size2);
    f << count << endl;
    cout << "# of Common Nodes: " << count << endl;
    f.close();

    // Delete objects
    for (int i = 0; i < size1; i++) {
        delete[] nodes1[i];
    }
    delete[] nodes1;
    for (int i = 0; i < size2; i++) {
        delete[] nodes2[i];
    }
    delete[] nodes2;

    return 0;
}
