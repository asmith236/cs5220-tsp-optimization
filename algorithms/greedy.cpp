#include <bits/stdc++.h>
#include "../common/algorithms.hpp"

using namespace std;

// Helper function to calculate Euclidean distance
double distance(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

TSPResult solve(const std::vector<std::pair<double, double>>& coordinates) {
    int n = coordinates.size();
    
    // Compute distance matrix
    vector<vector<double>> distances(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            distances[i][j] = distance(coordinates[i], coordinates[j]);
        }
    }

    // Initialize variables for greedy algorithm
    vector<bool> visited(n, false);
    vector<int> path;
    double total_cost = 0.0;
    
    // Start from city 0
    int current_city = 0;
    path.push_back(current_city);
    visited[current_city] = true;

    // Visit remaining n-1 cities
    while (path.size() < n) {
        double min_distance = numeric_limits<double>::infinity();
        int next_city = -1;

        // Find nearest unvisited city
        for (int j = 0; j < n; j++) {
            if (!visited[j] && distances[current_city][j] < min_distance) {
                min_distance = distances[current_city][j];
                next_city = j;
            }
        }

        // Add nearest city to path
        current_city = next_city;
        path.push_back(current_city);
        visited[current_city] = true;
        total_cost += min_distance;
    }

    // Add cost of returning to start
    total_cost += distances[path.back()][path[0]];

    TSPResult result;
    result.cost = total_cost;
    result.path = path;
    return result;
}