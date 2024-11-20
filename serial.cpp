#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <fstream>
#include <sstream>
#include "helper.hpp"

using namespace std;

// Constants
const int MAX = numeric_limits<int>::max(); // Use a large value as infinity

// Main function implementing Held-Karp
int heldKarp(vector<vector<int>>dist) {
    const int n = dist.size(); // Number of nodes in the input graph

    addExtraRowAndColumn(dist);
    printMatrix(dist);

    // DP table to store the cost of visiting subsets of nodes
    int dp[1 << n][n];

    // Initialize DP table
    for (int mask = 0; mask < (1 << n); mask++) {
        for (int i = 0; i < n; i++) {
            dp[mask][i] = MAX;
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
                    dp[mask][k] = min(dp[mask][k], dp[prevMask][m] + dist[m + 1][k + 1]);
                }
            }
        }
    }

    // Find the optimal tour cost
    int opt = MAX;
    int fullMask = (1 << n) - 1;
    for (int k = 1; k < n; k++) {
        opt = min(opt, dp[fullMask & ~(1 << 0)][k] + dist[k + 1][1]);
    }

    return opt;
}

int main() {
    // Solve TSP using Held-Karp
    vector<vector<int>> loadDist = loadMatrixFromCSV("dist_matrix.csv");
    
    int result = heldKarp(loadDist);
    cout << "The cost of the most efficient tour = " << result << endl;
    return 0;
}