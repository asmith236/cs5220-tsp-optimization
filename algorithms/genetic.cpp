#include <bits/stdc++.h>
#include "../common/algorithms.hpp" // Include the common header for TSPResult

using namespace std;

// Function to calculate the Euclidean distance between two points
double distance(const pair<double, double>& p1, const pair<double, double>& p2) {
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

// Function to compute the distance of a route
double routeDistance(const vector<int>& route, const vector<vector<double>>& distances) {
    double total_distance = 0.0;
    for (size_t i = 0; i < route.size() - 1; ++i) {
        total_distance += distances[route[i]][route[i + 1]];
    }
    total_distance += distances[route.back()][route.front()]; // Return to the starting point
    return total_distance;
}

// Fitness function (inverse of the route distance)
double fitness(const vector<int>& route, const vector<vector<double>>& distances) {
    double dist = routeDistance(route, distances);
    return dist == 0 ? 0 : 1.0 / dist; // Prevent divide-by-zero
}

// Function to create a random route
vector<int> createRoute(int n) {
    vector<int> route(n);
    iota(route.begin(), route.end(), 0); // Fill with 0, 1, ..., n-1
    random_device rd;
    mt19937 g(rd());
    shuffle(route.begin(), route.end(), g);
    return route;
}

// Function to initialize a population
vector<vector<int>> initialPopulation(int pop_size, int n) {
    vector<vector<int>> population(pop_size);
    for (int i = 0; i < pop_size; ++i) {
        population[i] = createRoute(n);
    }
    return population;
}

// Crossover function
vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2) {
    int n = parent1.size();
    vector<int> child(n, -1);
    random_device rd;
    mt19937 g(rd());
    uniform_int_distribution<> dis(0, n - 1);

    int start = dis(g), end = dis(g);
    if (start > end) swap(start, end);

    copy(parent1.begin() + start, parent1.begin() + end, child.begin() + start);
    auto it = child.begin();
    for (int city : parent2) {
        if (find(child.begin(), child.end(), city) == child.end()) {
            while (*it != -1) ++it;
            *it = city;
        }
    }
    return child;
}

// Mutation function
void mutate(vector<int>& route, double mutation_rate) {
    random_device rd;
    mt19937 g(rd());
    uniform_real_distribution<> prob(0, 1);
    uniform_int_distribution<> idx(0, route.size() - 1);

    for (size_t i = 0; i < route.size(); ++i) {
        if (prob(g) < mutation_rate) {
            int j = idx(g);
            swap(route[i], route[j]);
        }
    }
}

// Main genetic algorithm
TSPResult solve(const vector<pair<double, double>>& coordinates) {
    int n = coordinates.size();
    if (n < 2) {
        throw runtime_error("Error: The number of coordinates must be at least 2.");
    }

    // Create and initialize the distance matrix
    vector<vector<double>> distances(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            distances[i][j] = distance(coordinates[i], coordinates[j]);
        }
    }

    int pop_size = 100, elite_size = 20, generations = 300;
    double mutation_rate = 0.01;

    vector<vector<int>> population = initialPopulation(pop_size, n);
    vector<int> best_route;
    double best_distance = numeric_limits<double>::infinity();

    for (int gen = 0; gen < generations; ++gen) {
        // Rank population by fitness
        sort(population.begin(), population.end(), [&](const vector<int>& a, const vector<int>& b) {
            return fitness(a, distances) > fitness(b, distances);
        });

        // Select elite routes
        vector<vector<int>> parents(population.begin(), population.begin() + elite_size);
        vector<vector<int>> children;

        // Generate children using crossover and mutation
        random_device rd;
        mt19937 g(rd());
        for (int i = 0; i < pop_size - elite_size; ++i) {
            uniform_int_distribution<> dis(0, elite_size - 1);
            vector<int> child = crossover(parents[dis(g)], parents[dis(g)]);
            mutate(child, mutation_rate);
            children.push_back(child);
        }

        // Combine parents and children to form the next population
        population = parents;
        population.insert(population.end(), children.begin(), children.end());

        // Track the best route in this generation
        double current_best_distance = routeDistance(population[0], distances);
        if (current_best_distance < best_distance) {
            best_distance = current_best_distance;
            best_route = population[0];
        }
    }

    TSPResult result;
    result.cost = best_distance;
    result.path = best_route;
    return result;
}
