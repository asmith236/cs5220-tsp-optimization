#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <cassert>
#include "../common/algorithms.hpp" // Include the common header for TSPResult

using namespace std;

#define BLOCK_SIZE 256
#define DEBUG 0 // Debug flag for verbose output

// Device function to calculate the Euclidean distance between two points
__device__ double distance(const double2& p1, const double2& p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// Kernel to initialize the distance matrix
__global__ void initDistanceMatrix(const double2* coordinates, double* distances, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int i = idx / n;
        int j = idx % n;
        distances[idx] = distance(coordinates[i], coordinates[j]);
    }
}

// Kernel to calculate fitness values for a population
__global__ void calculateFitness(const int* population, const double* distances, double* fitness, int n, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        const int* route = &population[idx * n];
        double route_distance = 0.0;

        // Calculate the total distance of the route
        for (int i = 0; i < n - 1; ++i) {
            int from = route[i];
            int to = route[i + 1];
            route_distance += distances[from * n + to];
        }
        route_distance += distances[route[n - 1] * n + route[0]]; // Closing the loop

        // Avoid divide by zero
        if (route_distance <= 0.0) {
            fitness[idx] = 0.0;
        } else {
            fitness[idx] = 1.0 / route_distance;
        }

        // Debug: Print route distance
        if (DEBUG && threadIdx.x == 0) {
            printf("Route %d: Distance = %.6f, Fitness = %.6f\n", idx, route_distance, fitness[idx]);
        }
    }
}

// Device function to repair invalid routes
__device__ void rebuildRoute(int* route, int n) {
    bool visited[256] = {false};

    // Mark all valid and visited nodes
    for (int i = 0; i < n; ++i) {
        if (route[i] >= 0 && route[i] < n) {
            if (!visited[route[i]]) {
                visited[route[i]] = true;
            } else {
                route[i] = -1; // Mark duplicates as invalid
            }
        } else {
            route[i] = -1; // Mark invalid entries
        }
    }

    // Replace invalid or duplicate nodes
    int missing_idx = 0;
    for (int i = 0; i < n; ++i) {
        if (route[i] == -1) {
            while (missing_idx < n && visited[missing_idx]) {
                ++missing_idx;
            }
            route[i] = missing_idx;
            visited[missing_idx] = true;
        }
    }
}

// Kernel for mutation with optional debugging and repair
__global__ void mutatePopulation(int* population, int pop_size, int n, double mutation_rate, int seed, bool debug) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    if (idx < pop_size) {
        int* route = &population[idx * n];

        if (debug) {
            printf("Initial route for index %d: ", idx);
            for (int i = 0; i < n; ++i) {
                printf("%d ", route[i]);
            }
            printf("\n");
        }

        if (curand_uniform(&state) < mutation_rate) {
            int i = curand(&state) % n;
            int j = curand(&state) % n;
            while (j == i) {
                j = curand(&state) % n;
            }
            int temp = route[i];
            route[i] = route[j];
            route[j] = temp;
        }

        rebuildRoute(route, n);

        if (debug) {
            printf("Mutated and repaired route for index %d: ", idx);
            for (int i = 0; i < n; ++i) {
                printf("%d ", route[i]);
            }
            printf("\n");
        }
    }
}

// Function to generate random initial population on the host
vector<vector<int>> generateInitialPopulation(int pop_size, int n) {
    vector<vector<int>> population(pop_size, vector<int>(n));
    random_device rd;
    mt19937 g(rd());

    for (int i = 0; i < pop_size; ++i) {
        iota(population[i].begin(), population[i].end(), 0);
        shuffle(population[i].begin(), population[i].end(), g);
    }
    return population;
}

// Function to debug the initial population on the device
void debugInitialPopulationOnDevice(int* d_population, int pop_size, int n) {
    vector<int> host_population(pop_size * n);
    cudaMemcpy(host_population.data(), d_population, pop_size * n * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Initial population on device:" << endl;
    for (int i = 0; i < pop_size; ++i) {
        cout << "Route " << i << ": ";
        for (int j = 0; j < n; ++j) {
            cout << host_population[i * n + j] << " ";
        }
        cout << endl;
    }
}

// Main solve function using CUDA
TSPResult solve(const vector<pair<double, double>>& coordinates) {
    int n = coordinates.size();
    int pop_size = 100, generations = 300;
    double mutation_rate = 0.01;

    // Convert coordinates to double2
    vector<double2> host_coordinates(n);
    for (int i = 0; i < n; ++i) {
        host_coordinates[i] = {coordinates[i].first, coordinates[i].second};
    }

    // Allocate device memory
    double2* d_coordinates;
    double* d_distances;
    int* d_population;
    double* d_fitness;

    cudaMalloc(&d_coordinates, n * sizeof(double2));
    cudaMalloc(&d_distances, n * n * sizeof(double));
    cudaMalloc(&d_population, pop_size * n * sizeof(int));
    cudaMalloc(&d_fitness, pop_size * sizeof(double));

    cudaMemcpy(d_coordinates, host_coordinates.data(), n * sizeof(double2), cudaMemcpyHostToDevice);

    // Initialize distance matrix
    initDistanceMatrix<<<(n * n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_coordinates, d_distances, n);
    cudaDeviceSynchronize();

    // Generate and copy initial population
    auto host_population = generateInitialPopulation(pop_size, n);
    vector<int> flat_population;
    for (const auto& individual : host_population) {
        flat_population.insert(flat_population.end(), individual.begin(), individual.end());
    }
    cudaMemcpy(d_population, flat_population.data(), pop_size * n * sizeof(int), cudaMemcpyDeviceToHost);

    if (DEBUG) {
        debugInitialPopulationOnDevice(d_population, pop_size, n);
    }

    // Evolution loop
    vector<int> best_route;
    double best_distance = numeric_limits<double>::infinity();

    for (int gen = 0; gen < generations; ++gen) {
        calculateFitness<<<(pop_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_population, d_distances, d_fitness, n, pop_size);
        cudaDeviceSynchronize();

        thrust::device_ptr<double> fitness_ptr(d_fitness);
        thrust::device_ptr<int> population_ptr(d_population);

        // Ensure sorting is correct (lower distance -> higher fitness)
        thrust::sort_by_key(fitness_ptr, fitness_ptr + pop_size, population_ptr, thrust::greater<double>());

        mutatePopulation<<<(pop_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_population, pop_size, n, mutation_rate, gen, DEBUG);
        cudaDeviceSynchronize();

        if (DEBUG) {
            debugInitialPopulationOnDevice(d_population, pop_size, n);
        }

        vector<int> current_best_route(n);
        cudaMemcpy(current_best_route.data(), d_population, n * sizeof(int), cudaMemcpyDeviceToHost);

        double current_best_distance;
        cudaMemcpy(&current_best_distance, d_fitness, sizeof(double), cudaMemcpyDeviceToHost);

        double route_distance = 1.0 / current_best_distance; // Convert fitness back to distance
        if (route_distance < best_distance) {
            best_distance = route_distance;
            best_route = current_best_route;
        }

        if (DEBUG) {
            cout << "Generation " << gen << ": Best Distance = " << best_distance << endl;
        }
    }

    cudaFree(d_coordinates);
    cudaFree(d_distances);
    cudaFree(d_population);
    cudaFree(d_fitness);

    TSPResult result;
    result.cost = best_distance;
    result.path = best_route;
    return result;
}
