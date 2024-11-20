#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

using namespace std;

// Constants
const int n = 4; // Number of nodes in the graph
const int MAX = numeric_limits<int>::max(); // Use a large value as infinity

// Distance matrix representing the graph
int dist[n][n] = {
    { 0, 10, 15, 20 },
    { 10, 0, 35, 25 },
    { 15, 35, 0, 30 },
    { 20, 25, 30, 0 }
};

// DP table to store the cost of visiting subsets of nodes
int dp[1 << n][n];

// Table to track the previous node in the optimal path
int parent[1 << n][n];

// Structure to hold TSP result
struct TSPResult {
    int cost;
    vector<int> path;
};

// Function to reconstruct the tour path
vector<int> getPath(int fullMask, int lastNode) {
    vector<int> path;
    int mask = fullMask;

    // Reconstruct path backwards using the parent table
    while (lastNode != -1) {
        path.push_back(lastNode);
        int prevMask = mask & ~(1 << lastNode);
        lastNode = parent[mask][lastNode];
        mask = prevMask;
    }

    reverse(path.begin(), path.end()); // Reverse to get the correct order
    return path;
}

// Main function implementing Held-Karp
TSPResult heldKarp() {
    // Initialize DP table
    for (int mask = 0; mask < (1 << n); mask++) {
        for (int i = 0; i < n; i++) {
            dp[mask][i] = MAX;
            parent[mask][i] = -1;
        }
    }

    // Base case: cost to go from starting node (0) to every other node
    for (int k = 1; k < n; k++) {
        dp[1 << k][k] = dist[0][k];
        parent[1 << k][k] = 0;
    }

    // Iterate over subsets of size s
    for (int s = 2; s <= n; s++) {
        for (int mask = 0; mask < (1 << n); mask++) {
            if (__builtin_popcount(mask) != s) continue;

            for (int k = 0; k < n; k++) {
                if (!(mask & (1 << k))) continue;

                int prevMask = mask & ~(1 << k);
                for (int m = 0; m < n; m++) {
                    if (!(prevMask & (1 << m))) continue;

                    int newCost = dp[prevMask][m] + dist[m][k];
                    if (newCost < dp[mask][k]) {
                        dp[mask][k] = newCost;
                        parent[mask][k] = m;
                    }
                }
            }
        }
    }

    // Find the optimal tour cost
    int opt = MAX;
    int fullMask = (1 << n) - 1;
    int lastNode = -1;
    for (int k = 1; k < n; k++) {
        int cost = dp[fullMask & ~(1 << 0)][k] + dist[k][0];
        if (cost < opt) {
            opt = cost;
            lastNode = k;
        }
    }

    // Reconstruct the path
    vector<int> path = getPath(fullMask, lastNode);
    path.push_back(0); // Return to starting node

    return { opt, path };
}

int main(int argc, char* argv[]) {
    bool visualize = false;
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--viz") visualize = true;
    }

    TSPResult result = heldKarp();

    if (visualize) {
        // Output format for Python visualizer
        cout << n << endl; // Number of vertices
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << dist[i][j] << " ";
            }
            cout << endl;
        }
        cout << result.cost << endl;
        for (auto it = result.path.rbegin(); it != result.path.rend(); ++it) {
            cout << (*it + 1) << " "; // Convert 0-based to 1-based indexing
        }
        cout << endl;
    } else {
        // Print cost and path in default output
        cout << "The cost of the most efficient tour = " << result.cost << endl;
        cout << "The optimal tour path is: ";
        for (auto it = result.path.rbegin(); it != result.path.rend(); ++it) {
            cout << (*it + 1) << " "; // Convert 0-based to 1-based indexing
        }
        cout << endl;
    }

    return 0;
}