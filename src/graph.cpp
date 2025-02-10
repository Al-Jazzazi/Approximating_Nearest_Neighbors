#include "graph.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std; 

float termination_alpha = 0;
float termination_alpha2 = 0 ;
float bw_break = 0;



Graph::Graph(Config* config) {
    num_nodes = config->num_nodes;
    DIMENSION = config->dimensions;
    nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    mappings.resize(num_nodes);
    for (int i = 0; i < mappings.size(); i++) {
        mappings[i] = {};
    }
    start = 0;
    reset_statistics();
}


Graph::~Graph() {
    for (int i = 0; i < num_nodes; i++) {
        delete[] nodes[i];
    }
    delete[] nodes;
}


//load saved graph 
void Graph::load(Config* config) {
    
    ifstream graph_file(config->loaded_graph_file, std::ios::binary);
    cout << "Loading saved graph from " << config->loaded_graph_file << endl;
    if (!graph_file) {
        cout << "File " << config->loaded_graph_file << " not found!" << endl;
        return;
    }
    //Different import in case of nsg vs efanna 
    if( config->graph == "nsg"){
        graph_file.read((char *)&width, sizeof(unsigned));
        graph_file.read((char *)&start, sizeof(unsigned));
    }

    for (int i = 0; i < num_nodes; ++i) {
        unsigned num_neighbors;
        graph_file.read((char *)&num_neighbors, sizeof(unsigned));

        if (graph_file.eof()) break;

        for (unsigned j = 0; j < num_neighbors; ++j) {
            unsigned neighbor;
            graph_file.read((char *)&neighbor, sizeof(unsigned));
            mappings[i].emplace(neighbor);
        }
       
    }
    graph_file.close();
}
 


float Graph::find_distance(int i, float* query)  {
    ++distanceCalculationCount;
    return calculate_l2_sq(nodes[i], query, DIMENSION);
}

float Graph::find_distance(int i, int j) {
    ++distanceCalculationCount;
    return calculate_l2_sq(nodes[i], nodes[j], DIMENSION);
}



void Graph::reset_statistics(){
    distanceCalculationCount = 0;
    num_original_termination = 0; 
    num_distance_termination = 0;
    num_set_checks = 0; 
    size_of_c = 0;
    num_insertion_to_c = 0;
    num_deletion_from_c = 0;
    size_of_visited = 0;
 }





// Returns whether or not to terminate from search_layer
bool Graph::should_terminate(Config* config, priority_queue<pair<float, int>>& top_k, pair<float, int>& top_1, float close_squared, float far_squared,  int candidates_popped_per_q) {
    // Evaluate beam-width-based criteria
    bool beam_width_original = close_squared > far_squared;
    // Use candidates_popped as a proxy for beam-width
    bool beam_width_1 = candidates_popped_per_q > config->ef_search;
    bool beam_width_2 =  false;
    bool alpha_distance_1 = false;
    bool alpha_distance_2 = false;
    bool num_of_dist_1 = false;

    // Evaluate distance-based criteria
    if (config->use_hybrid_termination || config->use_distance_termination) {
        float close = sqrt(close_squared);
        float threshold = 2 * sqrt(top_k.top().first) + sqrt(top_1.first);
        // alpha * (2 * d_k + d_1)  --> 0 
        // alpha * 2 * d_k + d_1  --> 1 
        // alpha * (d_k + d_1)  + d_k --> 2 
        // alpha * d_1   + d_k   --> 3 
        // alpha * d_k   + d_k  --> 4 
        if(top_k.size() >= config->num_return && config->use_distance_termination)
            alpha_distance_1 =  config->alpha_termination_selection == 0 ? close > termination_alpha * (2 * sqrt(top_k.top().first) + sqrt(top_1.first)): 
                                config->alpha_termination_selection == 1 ? close > termination_alpha * 2 * sqrt(top_k.top().first) + sqrt(top_1.first): 
                                config->alpha_termination_selection == 2 ? close > termination_alpha *  (sqrt(top_k.top().first) + sqrt(top_1.first)) + sqrt(top_k.top().first) : 
                                config->alpha_termination_selection == 3 ? close > termination_alpha * sqrt(top_1.first) +  sqrt(top_k.top().first): 
                                                                           close > termination_alpha * sqrt(top_k.top().first) + sqrt(top_k.top().first);
        else{
            alpha_distance_1 = top_k.size() >= config->num_return && close > termination_alpha * threshold;

        }
      
        // Evaluate break points
        if (config->use_latest && config->use_break) {
            alpha_distance_2 = top_k.size() >= config->num_return && close > termination_alpha2 * threshold;
            beam_width_2 = candidates_popped_per_q > bw_break;

        }

    }
    // Return whether to terminate using config flags
    
    if (config->use_hybrid_termination && config->use_latest) {
        return (alpha_distance_1 && beam_width_1);
    } else if (config->use_hybrid_termination) {
        return alpha_distance_1 || beam_width_1;
    } else if (config->use_distance_termination) {
        return alpha_distance_1;
    } else if(config->use_calculation_termination) {
        return  config->calculations_per_query < distanceCalculationCount;
    } else {
        return beam_width_original;
    }
}



/*used to calculate percentiles in the cases where we want to 
1. run with a hybrid termination (we pass one termiantion and calculate the other through alrdy made graphs mapping alpha and beamWidthValue through the avg or median search distance calculations )
2. in case we want to have a break point based on the median values   
*/
void Graph::calculate_termination(Config *config){
        std::string alpha_key = std::to_string(config->num_return) + " " + config->dataset;
        cout << alpha_key << endl;

        config->alpha_coefficient = config->use_hybrid_termination ? config->alpha_vamana_values.at(alpha_key).first : 0;
        config->alpha_intercept = config->use_hybrid_termination ? config->alpha_vamana_values.at(alpha_key).second : 0;
        
        cout << "alpha_coefficient is " << config->alpha_coefficient << ", alpha_intercept is " << config->alpha_intercept <<endl; 
        
        float estimated_distance_calcs = config->bw_slope != 0 ? (config->ef_search - config->bw_intercept) / config->bw_slope : 1;
        termination_alpha = config->use_distance_termination ? config->termination_alpha : config->alpha_coefficient * log(estimated_distance_calcs) + config->alpha_intercept;
        
        cout << "estimated_distance_calcs is " << estimated_distance_calcs  << ", termination_alpha is " << termination_alpha << endl;
        estimated_distance_calcs *=config->alpha_break;
        termination_alpha2 = config->alpha_coefficient * log(estimated_distance_calcs) + config->alpha_intercept;
        
        
         ifstream histogram = ifstream(config->metric_prefix + "_median_percentiles.txt");
            //The part 
            if(!histogram.fail() && (config->use_median_break || config->use_median_earliast) ){
                vector<int> ef_search = {200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000,4500, 5000,5500, 6000, 6500, 7000, 7500, 8000, 8500}; 

                string info;
                int line  = find(ef_search.begin(),ef_search.end(), config->ef_search) - ef_search.begin();
                cout << "line is " << line << endl;
                int index = find(config->benchmark_median_percentiles.begin(),config->benchmark_median_percentiles.end(), config->breakMedian) - config->benchmark_median_percentiles.begin()+1; 
                int distance_termination = 0;
                while(line != 0 && getline(histogram,info)){
                    line--;
                }

                while(histogram >> info && index!= 0) {
                    index--;
            
                }
                estimated_distance_calcs = stoi(info);
            } 

        bw_break = static_cast<int>(config->bw_slope  * estimated_distance_calcs + config->bw_intercept);
        histogram.close();
        cout << "bw break is: " << bw_break << ", for estimated calc = " << estimated_distance_calcs; 
    }




void Graph::run_queries(Config* config, float** queries){
    vector<vector<int>> results;
    query(config, start, results, queries);
    vector<vector<int>> actualResults;
    get_actual_neighbors(config, actualResults, nodes, queries);
    int similar =0 ;
    cout << "results.size() " << results.size() <<  ", actualResults.size() " << actualResults.size() << endl ; 
    cout << "results[0].size() " << results[0].size() <<  ", actualResults[0].size() " << actualResults[0].size() << endl ; 
    cout << "results[0] " << results[0][0] <<  ", actualResults[0] " << actualResults[0][0] << endl ; 
    cout << "results[0] distance " << find_distance(results[0][0], queries[0]) <<  ", actualResults[0] distance " << find_distance(actualResults[0][0], queries[0]) << endl ; 

    for (int j = 0; j < config->num_queries; ++j) {
                
                unordered_set<int> actual_set(actualResults[j].begin(), actualResults[j].end());
                unordered_set<int> intersection;    
                for (int k = 0; k < results[j].size(); ++k) {
                    auto n_pair = results[j][k];

                    if (actual_set.find(n_pair) != actual_set.end()) {
                        intersection.insert(n_pair);
                    }
                }
                similar += intersection.size();
    
        }
    cout << "similar = " << similar << endl; 
    double recall = (double) similar / (config->num_queries * config->num_return);
    cout << "Recall of Graph is " << recall << endl;
}



void Graph::query(Config* config, int start, vector<vector<int>>& allResults, float** queries) {
   
    for (int k = 0; k < config->num_queries; k++) {
        if (k % 1000 == 0) cout << "Processing " << k << endl;
        float* thisQuery = queries[k];
        vector<int> result;
        beam_search(*this,config, start, thisQuery, config->ef_search, result);
        allResults.push_back(result);
    }
    cout << "All queries processed" << endl;
}


/*
Beam Search function that supports both beam width termination and distane based termination
*/
void beam_search(Graph& graph, Config* config,int start,  float* query, int bw, vector<int>& closest){
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> candidates;
    unordered_set<int> visited;
    priority_queue<pair<float, int>> found;
    priority_queue<pair<float, int>> top_k;
    pair<float, int> top_1;

    bool using_top_k = (config->use_hybrid_termination || config->use_distance_termination);

    //set bw to infinity (or a relatively large number)
    if (config->use_distance_termination || config->use_calculation_termination || config->use_hybrid_termination){
        bw = 100000;
    }

    float distance = graph.find_distance(start, query);
    candidates.emplace(make_pair(distance,start));      
    visited.emplace(start);
    found.emplace(make_pair(distance,start));

    graph.num_insertion_to_c++;  
    
    if (using_top_k) {
        top_k.emplace(make_pair(distance,start));
        top_1 = make_pair(distance,start);
    }

    int candidates_popped_per_q = 0;
    int iteration = 0;
    float far_dist = found.top().first;
    // cout << "first stop\n";
    while (!candidates.empty()) {
        far_dist = found.top().first;
        int closest = candidates.top().second;
        float close_dist = candidates.top().first;
        candidates.pop();
        
        ++candidates_popped_per_q;
        
        if (graph.should_terminate(config, top_k, top_1, close_dist, far_dist, candidates_popped_per_q)) {
            if(top_k.size() != config->num_return) cerr << "!!!!! top_k size does not matching number return!!!\n"; 
            if (config->use_hybrid_termination){
                if (candidates_popped_per_q > config->ef_search)
                    graph.num_original_termination++;
                else 
                    graph.num_distance_termination++;
            }
           
            break;
        }

        set<unsigned int>& neighbors = graph.mappings[closest];
        for (int neighbor : neighbors) {
            graph.num_set_checks++;
            if(visited.find(neighbor) == visited.end()){
                visited.insert(neighbor);
                float far_inner_dist = using_top_k? top_k.top().first : found.top().first;
                float neighbor_dist = graph.find_distance(neighbor,query);
                if ( (!using_top_k && (neighbor_dist < far_inner_dist || found.size() < bw)) ||
                        (using_top_k && (sqrt(neighbor_dist) <=  (1+ termination_alpha) * 2*sqrt(top_k.top().first)   ) ) ) {
                
                    candidates.emplace(make_pair(neighbor_dist, neighbor));

                    graph.num_insertion_to_c++;  
    
                    if (using_top_k) {
                        top_k.emplace(neighbor_dist, neighbor);
                        if (neighbor_dist < top_1.first) {
                            top_1 = make_pair(neighbor_dist, neighbor);
                        }
                        if (top_k.size() > config->num_return)
                            top_k.pop();
                    }
                    else{
                        found.emplace(neighbor_dist, neighbor);
                        if (found.size() > bw){
                            found.pop();
                        }
                    }
                }
            }

        }

    }
    // cout << "second stop\n";
    //Could've been added inside termination method
    graph.size_of_c += candidates.size(); 
    graph.num_deletion_from_c += candidates_popped_per_q; 
    graph.size_of_visited += visited.size();

    // cout << "thrid stop\n";
    int idx =using_top_k ? top_k.size() : found.size(); 
    closest.clear();
    closest.resize(idx);
    while (idx > 0) {
        --idx;
        if(using_top_k){
            closest[idx] = top_k.top().second;
            if(top_k.size() == 0) cerr << "errooor\n";
            top_k.pop();
        }
        else{
        closest[idx] = found.top().second;
        found.pop();
        }
    }
    // cout << "forth stop\n";
    closest.resize(min(closest.size(), (size_t)config->num_return));
    // cout << "fifth stop\n";
}

/*
Debugging functions below
*/
void Graph::print_100_mappings(Config* config){
    for(int i = 0; i<mappings.size(); i++){
        if(i ==100 ) break;
        cout << "i: " <<mappings[i].size() <<endl; 
    }
}

void Graph::print_avg_neigbor(Config* config){
    long long int sum  =0 ;
    for(int i = 0; i<mappings.size(); i++){
        sum += mappings[i].size();   
    }
    std::cout << "Avg # of neighbors  is = " << sum/mappings.size() << std::endl;
}

void Graph::print_k_neigbours(Config* config, int k){
    for(int i = 0; i<mappings.size(); i++){
        if(i ==k) break;
        cout << "i: " << i  << ", size of k is " << mappings[i].size() <<endl;
        for(auto n: mappings[i]){
            cout << n << " ";
        } 
       cout << endl;     
    }
}

void Graph::print_k_nodes( Config* config, int k){
    for(int i=0; i<config->num_nodes; i++){
        if(i ==k ) break;
        cout << "i: " << i << endl;
        for(int j =0; j< DIMENSION; j++){
            cout << ", " << nodes[i][j]; 
        }
        cout <<endl;
    }
}


