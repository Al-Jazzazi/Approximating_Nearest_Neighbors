#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "../config.h"
 
float calculate_l2_sq(float* a, float* b, int size);
void knn_search(Config* config, std::vector<std::vector<int>>& results, float** nodes, float** queries);
void load_fvecs(const std::string& file, float** results, int num, int dim, bool check_groundtruth = false);
void save_fvecs(const std::string& file, float** results, int num, int dim);
void load_ivecs(const std::string& file, std::vector<std::vector<int>>& results, int num, int dim);
void save_ivecs(const std::string& file, std::vector<std::vector<int>>& results);
void load_nodes(Config* config, float** nodes);
void load_queries(Config* config, float** nodes, float** queries);
void load_oracle(Config* config, std::vector<std::pair<int, int>>& result);
void get_actual_neighbors(Config* config, std::vector<std::vector<int>>& actual_neighbors, float** nodes, float** queries);
void find_similar(Config* config,  const std::vector<std::vector<int>> actual_neighbors, const std::vector<std::vector<int>> neighbors,  float** nodes, float** queries, int& similar, float& total_ndcg);
// std::string get_cpu_brand(); 

#endif