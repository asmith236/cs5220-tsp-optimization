#include <bits/stdc++.h>
#include "common/constants.hpp" // Include the common constants file
using namespace std;

struct TSPResult {
    int cost;
    vector<int> path;
};

TSPResult travellingSalesmanProblem(int graph[][n], int s) {
    vector<int> vertex;
    for (int i = 0; i < n; i++) {
        if (i != s) vertex.push_back(i);
    }

    TSPResult result;
    result.cost = MAX; // Use the MAX constant
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
    int s = 0;

    bool visualize = false;
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--viz") visualize = true;
    }

    TSPResult result = travellingSalesmanProblem(dist, s); // Use `dist` from constants.h

    if (visualize) {
        // Output format for Python visualizer
        cout << n << endl;  // Number of vertices
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << dist[i][j] << " ";
            }
            cout << endl;
        }
        cout << result.cost << endl;
        for (int v : result.path) {
            cout << v << " ";
        }
        cout << endl;
    } else {
        cout << result.cost << endl;
    }
    return 0;
}
