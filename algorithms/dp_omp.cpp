#include <bits/stdc++.h>
#include <omp.h> // Include OpenMP header for parallelization
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

    const int n = coordinates.size(); // Number of nodes

    // DP table to store the cost of visiting subsets of nodes
    std::vector<std::vector<double>> dp(1 << n, std::vector<double>(n, MAX));
    // Parent table to reconstruct the path
    std::vector<std::vector<int>> parent(1 << n, std::vector<int>(n, -1));

    // Base case: cost to go from node 0 to every other node
    #pragma omp parallel for schedule(static)
    for (int k = 1; k < n; k++) {
        dp[1 << k][k] = distance(coordinates[0], coordinates[k]);
    }

    // Iterate over subsets of size s
    for (int s = 2; s < n; s++) {
        #pragma omp parallel for schedule(dynamic, 64) // nowait
        for (int mask = 0; mask < (1 << n); mask++) {
            // Skip invalid subsets
            if (__builtin_popcount(mask) != s) continue;

            // Thread-local variables to avoid memory contention
            std::vector<double> local_dp(n, MAX);
            std::vector<int> local_parent(n, -1);

            // Iterate over all nodes in the subset
            for (int k = 1; k < n; k++) {
                if (!(mask & (1 << k))) continue; // Skip if k is not in the subset

                // Find the minimum cost to reach k from any other node m in the subset
                int prevMask = mask & ~(1 << k);
                double minCost = MAX; // Register variable for minimum cost
                int bestPrev = -1;

                for (int m = 1; m < n; m++) {
                    if (!(prevMask & (1 << m))) continue;
                    double newCost = dp[prevMask][m] + distance(coordinates[m], coordinates[k]);
                    if (newCost < minCost) {
                        minCost = newCost;
                        bestPrev = m;
                    }
                }

                // Update the thread-local storage
                local_dp[k] = minCost;
                local_parent[k] = bestPrev;
            }

            // Atomic updates to global DP table to avoid race conditions
            for (int k = 1; k < n; k++) {
                if (local_dp[k] < dp[mask][k]) {
                    #pragma omp atomic write
                    dp[mask][k] = local_dp[k];
                    #pragma omp atomic write
                    parent[mask][k] = local_parent[k];
                }
            }
        }
    }

    // Find the optimal tour cost and end node
    double opt = MAX;
    int lastNode = -1;
    int fullMask = (1 << n) - 1;
    for (int k = 1; k < n; k++) {
        double newCost = dp[fullMask & ~(1 << 0)][k] + distance(coordinates[k], coordinates[0]);
        if (newCost < opt) {
            opt = newCost;
            lastNode = k;
        }
    }

    // Reconstruct the path
    std::vector<int> best_path;
    int currentMask = fullMask & ~(1 << 0);
    int currentNode = lastNode;
    while (currentNode != -1) {
        best_path.push_back(currentNode);
        int temp = currentNode;
        currentNode = parent[currentMask][currentNode];
        currentMask &= ~(1 << temp);
    }
    best_path.push_back(0); // Add the starting node
    std::reverse(best_path.begin(), best_path.end());

    // Return the result
    TSPResult result;
    result.cost = opt;
    result.path = best_path;
    return result;

}