#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <vector>
#include <cpuid.h>
#include <fstream>
#include <string.h>
#include "vamana.h"

using namespace std;



int main() {
    time_t now = time(0);
    cout << "Benchmark run started at " << ctime(&now);
    RunSearch();
    now = time(0);
    cout << "Benchmark run ended at " << ctime(&now);
    
    
}


