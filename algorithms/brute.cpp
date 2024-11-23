#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // For std::next_permutation
#include <limits>    // For std::numeric_limits
#include "../common/algorithms.hpp" // Include the common constants file

using namespace std;

// Function to calculate the Euclidean distance between two points
double distance(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

TSPResult solve(const std::vector<std::pair<double, double>> &coordinates) {
    // should get the distances from some place

    int n = coordinates.size();

    // compute the distance matrix
    vector<vector<double>> distances(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            distances[i][j] = distance(coordinates[i], coordinates[j]);
        }
    }

    vector<int> cities(n);
    for (int i = 0; i < n; ++i) cities[i] = i; // Initialize cities as [0, 1, ..., n-1]

    double min_distance = numeric_limits<double>::infinity();
    vector<int> best_path;
    do {
        // Step 3: Calculate the total distance for the current permutation
        double total_distance = 0.0;
        for (int i = 0; i < n - 1; ++i) {
            total_distance += distances[cities[i]][cities[i + 1]];
        }
        total_distance += distances[cities[n - 1]][cities[0]]; // Return to the starting point

        // Step 4: Update the best path and minimum distance if needed
        if (total_distance < min_distance) {
            min_distance = total_distance;
            best_path = cities;
        }
    } while (next_permutation(cities.begin(), cities.end()));

    TSPResult result;
    result.cost = min_distance;
    result.path = best_path;
    return result;
}
