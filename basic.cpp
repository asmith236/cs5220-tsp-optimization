#include <bits/stdc++.h>
using namespace std;
#define V 4

struct TSPResult {
    int cost;
    vector<int> path;
};

TSPResult travellingSalesmanProblem(int graph[][V], int s) {
    vector<int> vertex;
    for(int i = 0; i < V; i++) {
        if (i != s) vertex.push_back(i);
    }

    TSPResult result;
    result.cost = INT_MAX;
    vector<int> best_path;

    do {
        int current_pathweight = 0;
        int k = s;
        vector<int> current_path = {s};  // Start with source

        for (size_t i = 0; i < vertex.size(); i++) {
            current_pathweight += graph[k][vertex[i]];
            k = vertex[i];
            current_path.push_back(k);
        }
        current_pathweight += graph[k][s];
        current_path.push_back(s);  // Return to source

        if (current_pathweight < result.cost) {
            result.cost = current_pathweight;
            result.path = current_path;
        }
    } while (next_permutation(vertex.begin(), vertex.end()));

    return result;
}

int main(int argc, char* argv[]) {
    int graph[][V] = {
        { 0, 10, 15, 20 },
        { 10, 0, 35, 25 },
        { 15, 35, 0, 30 },
        { 20, 25, 30, 0 }
    };
    int s = 0;

    bool visualize = false;
    for(int i = 1; i < argc; i++) {
        if(string(argv[i]) == "--viz") visualize = true;
    }

    TSPResult result = travellingSalesmanProblem(graph, s);

    if(visualize) {
        // Output format for Python visualizer
        cout << V << endl;  // Number of vertices
        for(int i = 0; i < V; i++) {
            for(int j = 0; j < V; j++) {
                cout << graph[i][j] << " ";
            }
            cout << endl;
        }
        cout << result.cost << endl;
        for(int v : result.path) {
            cout << v << " ";
        }
        cout << endl;
    } else {
        cout << result.cost << endl;
    }
    return 0;
}