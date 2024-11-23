#include <bits/stdc++.h>
using namespace std;

struct TSPResult {
    int cost;
    vector<int> path;
};

TSPResult travellingSalesmanProblem(const vector<vector<int>>& graph, int s) {
    int V = graph.size();
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
        vector<int> current_path = {s};

        for (size_t i = 0; i < vertex.size(); i++) {
            current_pathweight += graph[k][vertex[i]];
            k = vertex[i];
            current_path.push_back(k);
        }
        current_pathweight += graph[k][s];
        current_path.push_back(s);

        if (current_pathweight < result.cost) {
            result.cost = current_pathweight;
            result.path = current_path;
        }
    } while (next_permutation(vertex.begin(), vertex.end()));

    return result;
}

vector<vector<int>> readGraphFromFile(const string& filename) {
    ifstream file(filename);
    int V;
    file >> V;
    
    vector<vector<int>> graph(V, vector<int>(V));
    for(int i = 0; i < V; i++) {
        for(int j = 0; j < V; j++) {
            file >> graph[i][j];
        }
    }
    return graph;
}

int main(int argc, char* argv[]) {
    vector<vector<int>> graph;
    
    if(argc > 1) {
        // Read from file if filename provided
        graph = readGraphFromFile(argv[1]);
    } else {
        // Default 4x4 graph if no file provided
        graph = {
            { 0, 10, 15, 20 },
            { 10, 0, 35, 25 },
            { 15, 35, 0, 30 },
            { 20, 25, 30, 0 }
        };
    }

    int s = 0;
    bool visualize = false;
    for(int i = 1; i < argc; i++) {
        if(string(argv[i]) == "--viz") visualize = true;
    }

    TSPResult result = travellingSalesmanProblem(graph, s);

    if(visualize) {
        cout << graph.size() << endl;
        for(const auto& row : graph) {
            for(int val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
        cout << result.cost << endl;
        for(int v : result.path) {
            cout << v << " ";
        }
        cout << endl;
    } else {
        cout << "Cost: " << result.cost << endl;
        cout << "Path: ";
        for(int v : result.path) {
            cout << v << " ";
        }
        cout << endl;
    }
    return 0;
}