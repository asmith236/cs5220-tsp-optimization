#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>
#include "../common/algorithms.hpp"

using namespace std;

#define BLOCK_SIZE 256
#define DEBUG 0

// ---------------- Device kernels and functions ----------------

__device__ double distance_dev(const double2& p1, const double2& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

__global__ void initDistanceMatrix(const double2* coordinates, double* distances, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int i = idx / n;
        int j = idx % n;
        distances[idx] = distance_dev(coordinates[i], coordinates[j]);
    }
}

__global__ void calculateFitness(const int* population, const double* distances, double* fitness, int n, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        const int* route = &population[idx * n];
        double route_distance = 0.0;
        for (int i = 0; i < n - 1; ++i) {
            int from = route[i];
            int to = route[i + 1];
            route_distance += distances[from * n + to];
        }
        route_distance += distances[route[n - 1] * n + route[0]];
        fitness[idx] = (route_distance > 0.0) ? 1.0 / route_distance : 0.0;
    }
}

__device__ void rebuildRoute(int* route, int n) {
    bool visited[256] = {false};
    for (int i = 0; i < n; ++i) {
        if (route[i] >= 0 && route[i] < n) {
            if (!visited[route[i]]) {
                visited[route[i]] = true;
            } else {
                route[i] = -1;
            }
        } else {
            route[i] = -1;
        }
    }
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

__global__ void mutatePopulation(int* population, int pop_size, int n, double mutation_rate, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        int* route = &population[idx * n];
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
    }
}

// Simple Order Crossover on the device
__global__ void crossoverOX(const int* parents, int* offspring, int n, int half, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half / 2) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);

        int parent1_idx = idx * 2;
        int parent2_idx = idx * 2 + 1;

        const int* p1 = &parents[parent1_idx * n];
        const int* p2 = &parents[parent2_idx * n];

        int cut1 = curand(&state) % n;
        int cut2 = curand(&state) % n;
        if (cut2 < cut1) {
            int temp = cut1;
            cut1 = cut2;
            cut2 = temp;
        }

        int* o1 = &offspring[parent1_idx * n];
        int* o2 = &offspring[parent2_idx * n];

        for (int i = 0; i < n; i++) {
            o1[i] = -1;
            o2[i] = -1;
        }

        for (int i = cut1; i <= cut2; i++) {
            o1[i] = p1[i];
            o2[i] = p2[i];
        }

        // Fill o1 from p2
        int current = 0;
        for (int i = 0; i < n; i++) {
            int city = p2[i];
            bool found = false;
            for (int j = cut1; j <= cut2; j++) {
                if (o1[j] == city) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                while (o1[current] != -1) {
                    current++;
                    if (current >= n) current = 0;
                }
                o1[current] = city;
            }
        }

        // Fill o2 from p1
        current = 0;
        for (int i = 0; i < n; i++) {
            int city = p1[i];
            bool found = false;
            for (int j = cut1; j <= cut2; j++) {
                if (o2[j] == city) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                while (o2[current] != -1) {
                    current++;
                    if (current >= n) current = 0;
                }
                o2[current] = city;
            }
        }

        // repair
        // Not strictly needed if we trust OX, but let's be safe:
        rebuildRoute(o1, n);
        rebuildRoute(o2, n);
    }
}

// ---------------- Host-side helper functions ----------------

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

TSPResult solve(const vector<pair<double, double>>& coordinates) {
    int n = (int)coordinates.size();
    int generations = 600;
    double mutation_rate = 0.005;
    int pop_size = 100;

    if(n % 2 != 0) {
        pop_size = n-1;
    }
    else{
        pop_size = n;
    }

    vector<double2> host_coordinates(n);
    for (int i = 0; i < n; ++i) {
        host_coordinates[i] = {coordinates[i].first, coordinates[i].second};
    }

    double2* d_coordinates;
    double* d_distances;
    int* d_population;
    double* d_fitness;

    cudaMalloc(&d_coordinates, n * sizeof(double2));
    cudaMalloc(&d_distances, n * n * sizeof(double));
    cudaMalloc(&d_population, pop_size * n * sizeof(int));
    cudaMalloc(&d_fitness, pop_size * sizeof(double));

    cudaMemcpy(d_coordinates, host_coordinates.data(), n * sizeof(double2), cudaMemcpyHostToDevice);
    initDistanceMatrix<<<(n*n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_coordinates, d_distances, n);
    cudaDeviceSynchronize();

    auto host_population = generateInitialPopulation(pop_size, n);
    vector<int> flat_population(pop_size * n);
    for (int i = 0; i < pop_size; i++) {
        for (int j = 0; j < n; j++) {
            flat_population[i*n + j] = host_population[i][j];
        }
    }

    cudaMemcpy(d_population, flat_population.data(), pop_size*n*sizeof(int), cudaMemcpyHostToDevice);

    int half = pop_size / 2;
    int* d_parents;
    int* d_offspring;
    cudaMalloc(&d_parents, half*n*sizeof(int));
    cudaMalloc(&d_offspring, half*n*sizeof(int));

    vector<int> host_pop(pop_size * n);
    vector<double> host_fitness(pop_size);
    vector<int> best_route;
    double best_distance = numeric_limits<double>::infinity();

    for (int gen = 0; gen < generations; ++gen) {
        // Compute fitness
        calculateFitness<<<(pop_size + BLOCK_SIZE - 1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_population, d_distances, d_fitness, n, pop_size);
        cudaDeviceSynchronize();

        // Copy fitness and population to host
        cudaMemcpy(host_fitness.data(), d_fitness, pop_size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_pop.data(), d_population, pop_size*n*sizeof(int), cudaMemcpyDeviceToHost);

        // Sort on host by fitness
        vector<int> indices(pop_size);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&](int a, int b) {
            return host_fitness[a] > host_fitness[b];
        });

        // Reorder population on host according to fitness
        {
            vector<int> sorted_pop(pop_size * n);
            for (int i = 0; i < pop_size; i++) {
                int idx = indices[i];
                for (int j = 0; j < n; j++) {
                    sorted_pop[i*n + j] = host_pop[idx*n + j];
                }
            }
            host_pop = sorted_pop;
        }

        // Find best route on host
        {
            double best_fitness_this_gen = host_fitness[indices[0]];
            double current_best_distance = (best_fitness_this_gen > 0.0) ? 1.0 / best_fitness_this_gen : numeric_limits<double>::infinity();
            if (current_best_distance < best_distance) {
                best_distance = current_best_distance;
                best_route.assign(host_pop.begin(), host_pop.begin() + n);
            }
        }

        // Selection: top half as parents
        // Copy top half back to device (parents)
        cudaMemcpy(d_parents, host_pop.data(), half*n*sizeof(int), cudaMemcpyHostToDevice);

        // Crossover on device (produce offspring)
        crossoverOX<<<(half/2+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_parents, d_offspring, n, half, gen);
        cudaDeviceSynchronize();

        // Place parents + offspring back into host_pop
        // Top half already parents, bottom half replaced by offspring
        cudaMemcpy(host_pop.data(), d_parents, half*n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_pop.data() + half*n, d_offspring, half*n*sizeof(int), cudaMemcpyDeviceToHost);

        // Copy combined population back to device
        cudaMemcpy(d_population, host_pop.data(), pop_size*n*sizeof(int), cudaMemcpyHostToDevice);

        // Mutation
        mutatePopulation<<<(pop_size+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_population, pop_size, n, mutation_rate, gen);
        cudaDeviceSynchronize();

        if (DEBUG && gen % 10 == 0) {
            cout << "Gen " << gen << " best distance = " << best_distance << endl;
        }
    }

    cudaFree(d_coordinates);
    cudaFree(d_distances);
    cudaFree(d_population);
    cudaFree(d_fitness);
    cudaFree(d_parents);
    cudaFree(d_offspring);

    TSPResult result;
    result.cost = best_distance;
    result.path = best_route;
    return result;
}
