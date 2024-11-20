#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

using namespace std;

// Constants
const int n = 4; // Number of nodes in the graph
const int MAX = numeric_limits<int>::max(); // Use a large value as infinity

// Distance matrix representing the graph
int dist[n + 1][n + 1] = {
    {0, 0, 0, 0, 0},    {0, 0, 10, 15, 20},
    {0, 10, 0, 25, 25}, {0, 15, 25, 0, 30},
    {0, 20, 25, 30, 0},
};

// DP table to store the cost of visiting subsets of nodes
int dp[1 << n][n];
int parent[1 << n][n]; // Table to store the parent of each node for backtracking

// Main function implementing Held-Karp
pair<int, vector<int>> heldKarp() {
    // Initialize DP table and parent table
    for (int mask = 0; mask < (1 << n); mask++) {
        for (int i = 0; i < n; i++) {
            dp[mask][i] = MAX;
            parent[mask][i] = -1;
        }
    }

    // Base case: cost to go from 1 to every other node
    for (int k = 1; k < n; k++) {
        dp[1 << k][k] = dist[1][k + 1]; // Convert 1-based to 0-based
    }

    // Iterate over subsets of size s
    for (int s = 2; s < n; s++) {
        for (int mask = 0; mask < (1 << n); mask++) {
            // Skip invalid subsets
            if (__builtin_popcount(mask) != s) continue;

            // Iterate over all nodes in the subset
            for (int k = 1; k < n; k++) {
                if (!(mask & (1 << k))) continue; // Skip if k is not in the subset

                // Find the minimum cost to reach k from any other node m in the subset
                int prevMask = mask & ~(1 << k);
                for (int m = 1; m < n; m++) {
                    if (!(prevMask & (1 << m))) continue;
                    int newCost = dp[prevMask][m] + dist[m + 1][k + 1];
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
    int lastNode = -1;
    int fullMask = (1 << n) - 1;
    for (int k = 1; k < n; k++) {
        int tourCost = dp[fullMask & ~(1 << 0)][k] + dist[k + 1][1];
        if (tourCost < opt) {
            opt = tourCost;
            lastNode = k;
        }
    }

    // Backtrack to find the tour path
    vector<int> path;
    int mask = (1 << n) - 1;
    int currentNode = lastNode;
    while (currentNode != -1) {
        path.push_back(currentNode + 1); // Convert 0-based to 1-based
        int nextMask = mask & ~(1 << currentNode);
        currentNode = parent[mask][currentNode];
        mask = nextMask;
    }

    // Add the starting node to complete the cycle
    path.push_back(1);
    reverse(path.begin(), path.end());

    return {opt, path};
}

int main() {
    // Solve TSP using Held-Karp
    auto result = heldKarp();
    int cost = result.first;
    vector<int> path = result.second;

    cout << "The cost of the most efficient tour = " << cost << endl;
    cout << "The optimal tour path is: ";
    for (int node : path) {
        cout << node << " ";
    }
    cout << endl;

    return 0;
}
