#include <iostream>
#include <vector>

#include "../include/utils.h"

using namespace std;


/*
In this file we compare whether all the datasets in a given 
*/
bool compare_node_query(Config* config, float* node, float* query){
    
    for(int i = 0; i<config->dimensions; i++){
      if(*(node+i) != *(query+i)) return false;  
    }
    return true;
}


void compare_nodes_queries(Config* config, float** nodes, float** queries){
    int num_similarties = 0; 
    for(int i = 0; i< config->num_nodes; i++){
        for(int j = 0; j < config->num_queries; j++){
            if(compare_node_query(config, nodes[i], queries[j]))  num_similarties++;
        }
    }

    cout << "num of similarities is " << num_similarties; 

}


int main() {
    // Load config
    Config* config = new Config();

    // Load nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    float** queries = new float*[config->num_queries];
    load_queries(config, nodes, queries);

    compare_nodes_queries(config, nodes, queries);


}