#include <iostream>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>

#include "../include/config.h"

using namespace std;
using namespace std::chrono;

template<typename T> 
void load_container(T& container, string file_name) {
    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Error opening candidates file!" << endl;
        return;
    }

    string line;
    int prev_queryID; 
    getline(file, line);

    while (getline(file, line)) {
    stringstream ss(line);
    string operation;  
    int queryID; 
    int node;          
    float dist;        
    char comma;

    if (ss >> queryID >> comma >> operation >> comma >> node >> comma >> dist) {
        if(queryID != prev_queryID){
            T container2; 
            container = container2;
        }
        else if (operation == "push") {  // Only store pushes in candidates
            container.emplace(dist, node);
        }
        else{
            container.pop();
        }
        prev_queryID = queryID; 
    }
    }

    file.close();
}



void load_container( unordered_set<int>& visited, string file_name) {
    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Error opening candidates file!" << endl;
        return;
    }

    string line;
    int prev_queryID; 
    getline(file, line);

    while (getline(file, line)) {
    stringstream ss(line);
    string operation;  
    int queryID; 
    int node;          
    float dist;        
    char comma;

    if (ss >> queryID >> comma >> operation >> comma >> node >> comma >> dist) {
        if(queryID != prev_queryID)
            visited.clear(); 
        if (operation == "push") {  // Only store pushes in candidates
            visited.insert(node);
        }

        prev_queryID = queryID; 
    }
    }

    file.close();
}





int main(){
    Config* config = new Config();


    const string candidates_file = "./data_eval/candidates/" + config->graph +"/_" +config->dataset + "_k=" + std::to_string(config->num_return) + "_distance_term_" + std::to_string(config->alpha_termination_selection)  + ".csv";
    const string found_file = "./data_eval/found/" + config->graph +"/_" +config->dataset + "_k=" + std::to_string(config->num_return) + "_distance_term_" + std::to_string(config->alpha_termination_selection)  + ".csv";
    const string visited_file = "./data_eval/visited/" + config->graph +"/_" +config->dataset + "_k=" + std::to_string(config->num_return) + "_distance_term_" + std::to_string(config->alpha_termination_selection)  + ".csv";

    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> candidates;
    priority_queue<pair<float, int>> found;
    unordered_set<int> visited;



     auto start = high_resolution_clock::now();
    load_container(candidates, candidates_file);
    load_container(found, found_file);
    load_container(visited, visited_file);
    auto end = high_resolution_clock::now();
    cout << "Time to load :" << duration_cast<milliseconds>(end - start).count() << " ms\n";





}