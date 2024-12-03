#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // For std::next_permutation
#include <limits>    // For std::numeric_limits
#include "../common/algorithms.hpp" // Include the common constants file

using namespace std;

const int MAX = numeric_limits<int>::max(); // Use a large value as infinity

double distance(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

TSPResult solve(const std::vector<std::pair<double, double>> &coordinates) {

    int n = coordinates.size();
    vector<vector<double>> distances(n, vector<double>(n));

    // Compute all pairwise distances
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            distances[i][j] = distance(coordinates[i], coordinates[j]);
        }
    }

    // dp[mask][i] stores the minimum cost to visit all nodes in `mask` and end at node `i`
    vector<vector<double>> dp(1 << n, vector<double>(n, INT_MAX));
    dp[1][0] = 0;  // Base case: starting at node 0

    // Fill the dp table
    for (int mask = 0; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if ((mask & (1 << u)) == 0) continue;  // Skip if node `u` is not visited in this mask
            for (int v = 0; v < n; v++) {
                if ((mask & (1 << v)) == 0) continue;  // Skip if node `v` is not visited in this mask
                dp[mask][u] = min(dp[mask][u], dp[mask ^ (1 << u)][v] + distances[v][u]);
            }
        }
    }

    printf("DP:\n");
    for (int mask = 0; mask < (1 << n); mask++) {  // Iterate through all subsets (2^n subsets)
        for (int u = 0; u < n; u++) {  // Iterate through each node in the subset
            printf("%f ", dp[mask][u]);  // Access and print the value for the current subset and node
        }
        printf("\n");  // Move to the next line after each subset
    }
    // Find the minimum cost to complete the tour and return to the starting point (node 0)
    double min_distance = INT_MAX;
    for (int u = 1; u < n; u++) {
        min_distance = min(min_distance, dp[(1 << n) - 1][u] + distances[u][0]);
    }

    // Reconstruct the best path
    int mask = (1 << n) - 1;  // All nodes visited
    int last = 0;  // Start at node 0
    vector<int> best_path = {0};

    for (int i = 1; i < n; i++) {
        int next_city = -1;
        double min_cost = INT_MAX;
        
        for (int v = 0; v < n; v++) {
            if (mask & (1 << v)) {  // If node `v` is still unvisited in this mask
                double cost = dp[mask][v] + distances[v][last];
                if (cost < min_cost) {
                    min_cost = cost;
                    next_city = v;
                }
            }
        }

        best_path.push_back(next_city);
        mask ^= (1 << next_city);  // Mark node `next_city` as visited
        last = next_city;
    }

    best_path.push_back(0);  // Return to the starting node
    
    // Return the result as TSPResult struct
    return TSPResult{min_distance, best_path};

}