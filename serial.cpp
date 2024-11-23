#include <bits/stdc++.h>
using namespace std;

struct TSPResult {
    int cost;
    vector<int> path;
};

// Function to read graph from file
vector<vector<int>> readGraphFromFile(const string& filename) {
    ifstream file(filename);
    int n;
    file >> n;
    
    vector<vector<int>> dist(n, vector<int>(n));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            file >> dist[i][j];
        }
    }
    return dist;
}

// Modified Held-Karp to use vector instead of fixed arrays
TSPResult heldKarp(const vector<vector<int>>& dist) {
    int n = dist.size();
    vector<vector<int>> dp(1 << n, vector<int>(n, INT_MAX));
    vector<vector<int>> parent(1 << n, vector<int>(n, -1));

    // Base case
    for(int k = 1; k < n; k++) {
        dp[1 << k][k] = dist[0][k];
        parent[1 << k][k] = 0;
    }

    // Iterate over subsets
    for(int s = 2; s <= n; s++) {
        for(int mask = 0; mask < (1 << n); mask++) {
            if(__builtin_popcount(mask) != s) continue;

            for(int k = 0; k < n; k++) {
                if(!(mask & (1 << k))) continue;

                int prevMask = mask & ~(1 << k);
                for(int m = 0; m < n; m++) {
                    if(!(prevMask & (1 << m))) continue;

                    int newCost = dp[prevMask][m] + dist[m][k];
                    if(newCost < dp[mask][k]) {
                        dp[mask][k] = newCost;
                        parent[mask][k] = m;
                    }
                }
            }
        }
    }

    // Find optimal tour
    int opt = INT_MAX;
    int fullMask = (1 << n) - 1;
    int lastNode = -1;
    for(int k = 1; k < n; k++) {
        int cost = dp[fullMask & ~(1 << 0)][k] + dist[k][0];
        if(cost < opt) {
            opt = cost;
            lastNode = k;
        }
    }

    // Reconstruct path
    vector<int> path;
    int mask = fullMask;
    while(lastNode != -1) {
        path.push_back(lastNode);
        int prevMask = mask & ~(1 << lastNode);
        lastNode = parent[mask][lastNode];
        mask = prevMask;
    }
    reverse(path.begin(), path.end());
    path.push_back(0);

    return {opt, path};
}

int main(int argc, char* argv[]) {
    vector<vector<int>> dist;
    bool visualize = false;

    if(argc > 1) {
        dist = readGraphFromFile(argv[1]);
        visualize = (argc > 2 && string(argv[2]) == "--viz");
    } else {
        // Default 4x4 matrix if no input file provided
        dist = {
            { 0, 10, 15, 20 },
            { 10, 0, 35, 25 },
            { 15, 35, 0, 30 },
            { 20, 25, 30, 0 }
        };
    }

    TSPResult result = heldKarp(dist);

    if(visualize) {
        cout << dist.size() << endl;
        for(const auto& row : dist) {
            for(int val : row) cout << val << " ";
            cout << endl;
        }
        cout << result.cost << endl;
        for(int v : result.path) cout << v << " ";
        cout << endl;
    } else {
        cout << "Cost: " << result.cost << endl;
        cout << "Path: ";
        for(int v : result.path) cout << v << " ";
        cout << endl;
    }

    return 0;
}