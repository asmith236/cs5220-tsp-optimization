// TINY: ~27
// SMALL: ~310
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include "../common/algorithms.hpp" // Include the common header for TSPResult

using namespace std;

#define BLOCK_SIZE 256

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
        if (i < n && j < n) { // Safe bounds
            distances[idx] = distance(coordinates[i], coordinates[j]);
        }
    }
}

// Kernel to calculate fitness values for a population
__global__ void calculateFitness(const int* population, const double* distances, double* fitness, int n, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        const int* route = &population[idx * n];
        double route_distance = 0.0;

        for (int i = 0; i < n - 1; ++i) {
            int from = route[i];
            int to = route[i + 1];
            if (from >= 0 && from < n && to >= 0 && to < n) {
                route_distance += distances[from * n + to];
            } else {
                printf("Invalid indices in fitness calculation: from=%d, to=%d, idx=%d, n=%d\n", from, to, idx, n);
            }
        }
        int last = route[n - 1];
        int first = route[0];
        if (last >= 0 && last < n && first >= 0 && first < n) {
            route_distance += distances[last * n + first];
        } else {
            printf("Invalid indices in fitness calculation: last=%d, first=%d, idx=%d, n=%d\n", last, first, idx, n);
        }
        fitness[idx] = (route_distance > 0) ? 1.0 / route_distance : 0.0;
    }
}


// Kernel for mutation
__global__ void mutatePopulation(int* population, int n, double mutation_rate, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    if (idx < n) {
        int* route = &population[idx * n];
        for (int i = 0; i < n; ++i) {
            if (curand_uniform(&state) < mutation_rate) {
                int j = curand(&state) % n;
                // Swap cities
                int temp = route[i];
                route[i] = route[j];
                route[j] = temp;
            }
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

    // Copy coordinates to device
    cudaMemcpy(d_coordinates, host_coordinates.data(), n * sizeof(double2), cudaMemcpyHostToDevice);

    // Initialize distance matrix
    initDistanceMatrix<<<(n * n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_coordinates, d_distances, n);
    cudaDeviceSynchronize();

    // Debug: Print distance matrix
    vector<double> host_distances(n * n);
    cudaMemcpy(host_distances.data(), d_distances, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    cout << "Distance matrix:" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << host_distances[i * n + j] << " ";
        }
        cout << endl;
    }

    // Generate initial population on the host
    auto host_population = generateInitialPopulation(pop_size, n);

    // Debug: Print generated host population
    cout << "Generated host population:" << endl;
    for (const auto& individual : host_population) {
        for (int gene : individual) {
            cout << gene << " ";
        }
        cout << endl;
    }

    // Flatten the population for memory copy
    vector<int> flat_population;
    for (const auto& individual : host_population) {
        flat_population.insert(flat_population.end(), individual.begin(), individual.end());
    }

    // Copy population to device
    cudaMemcpy(d_population, flat_population.data(), pop_size * n * sizeof(int), cudaMemcpyHostToDevice);

    // Debug: Validate copied population
    vector<int> host_population_flat(pop_size * n);
    cudaMemcpy(host_population_flat.data(), d_population, pop_size * n * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Copied population to device:" << endl;
    for (int i = 0; i < pop_size; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << host_population_flat[i * n + j] << " ";
        }
        cout << endl;
    }

    // Evolution loop
    vector<int> best_route;
    double best_distance = numeric_limits<double>::infinity();

    for (int gen = 0; gen < generations; ++gen) {
        // Calculate fitness
        calculateFitness<<<(pop_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_distances, d_fitness, n, pop_size);
        cudaDeviceSynchronize();

        // Debug: Print fitness values
        vector<double> host_fitness(pop_size);
        cudaMemcpy(host_fitness.data(), d_fitness, pop_size * sizeof(double), cudaMemcpyDeviceToHost);

        cout << "Generation " << gen << " fitness: ";
        for (double f : host_fitness) {
            cout << f << " ";
        }
        cout << endl;

        // Sort population by fitness (using Thrust)
        thrust::device_ptr<double> fitness_ptr(d_fitness);
        thrust::device_ptr<int> population_ptr(d_population);
        thrust::sort_by_key(fitness_ptr, fitness_ptr + pop_size, population_ptr);

        // Mutate population
        mutatePopulation<<<(pop_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, n, mutation_rate, gen);
        cudaDeviceSynchronize();

        // Copy best individual back to host
        vector<int> current_best_route(n);
        cudaMemcpy(current_best_route.data(), d_population, n * sizeof(int), cudaMemcpyDeviceToHost);

        double current_best_distance;
        cudaMemcpy(&current_best_distance, d_fitness, sizeof(double), cudaMemcpyDeviceToHost);

        if (1.0 / current_best_distance < best_distance) {
            best_distance = 1.0 / current_best_distance;
            best_route = current_best_route;
        }
    }

    // Free device memory
    cudaFree(d_coordinates);
    cudaFree(d_distances);
    cudaFree(d_population);
    cudaFree(d_fitness);

    // Return the best result
    TSPResult result;
    result.cost = best_distance;
    result.path = best_route;
    return result;
}











// FULLY BROKEN ===============================================================
// // TINY: ~27
// // SMALL: ~310
// #include <cuda_runtime.h>
// #include <curand_kernel.h>
// #include <thrust/device_vector.h>
// #include <thrust/sort.h>
// #include <vector>
// #include <iostream>
// #include <numeric>
// #include <random>
// #include "../common/algorithms.hpp" // Include the common header for TSPResult

// using namespace std;

// #define BLOCK_SIZE 256

// // Device function to calculate the Euclidean distance between two points
// __device__ double distance(const double2& p1, const double2& p2) {
//     return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
// }

// // Kernel to initialize the distance matrix
// __global__ void initDistanceMatrix(const double2* coordinates, double* distances, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n * n) {
//         int i = idx / n;
//         int j = idx % n;
//         if (i < n && j < n) { // Safe bounds
//             distances[idx] = distance(coordinates[i], coordinates[j]);
//         }
//     }
// }

// // Kernel to calculate fitness values for a population
// __global__ void calculateFitness(const int* population, const double* distances, double* fitness, int n, int pop_size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < pop_size) {
//         const int* route = &population[idx * n];
//         double route_distance = 0.0;

//         for (int i = 0; i < n - 1; ++i) {
//             int from = route[i];
//             int to = route[i + 1];
//             if (from >= 0 && from < n && to >= 0 && to < n) {
//                 route_distance += distances[from * n + to];
//             } else {
//                 printf("Invalid indices in fitness calculation: from=%d, to=%d, idx=%d, n=%d\n", from, to, idx, n);
//             }
//         }
//         int last = route[n - 1];
//         int first = route[0];
//         if (last >= 0 && last < n && first >= 0 && first < n) {
//             route_distance += distances[last * n + first];
//         } else {
//             printf("Invalid indices in fitness calculation: last=%d, first=%d, idx=%d, n=%d\n", last, first, idx, n);
//         }
//         fitness[idx] = (route_distance > 0) ? 1.0 / route_distance : 0.0;
//     }
// }


// // Kernel for mutation
// __global__ void mutatePopulation(int* population, int n, double mutation_rate, int seed) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     curandState state;
//     curand_init(seed + idx, 0, 0, &state);

//     if (idx < n) {
//         int* route = &population[idx * n];
//         for (int i = 0; i < n; ++i) {
//             if (curand_uniform(&state) < mutation_rate) {
//                 int j = curand(&state) % n;
//                 // Swap cities
//                 int temp = route[i];
//                 route[i] = route[j];
//                 route[j] = temp;
//             }
//         }
//     }
// }

// // Function to generate random initial population on the host
// vector<vector<int>> generateInitialPopulation(int pop_size, int n) {
//     vector<vector<int>> population(pop_size, vector<int>(n));
//     random_device rd;
//     mt19937 g(rd());

//     for (int i = 0; i < pop_size; ++i) {
//         iota(population[i].begin(), population[i].end(), 0);
//         shuffle(population[i].begin(), population[i].end(), g);
//     }
//     return population;
// }

// vector<int> fixRoute(const vector<int>& route, int n) {
//     vector<bool> visited(n, false);
//     vector<int> fixed_route;
//     for (int city : route) {
//         if (!visited[city]) {
//             fixed_route.push_back(city);
//             visited[city] = true;
//         }
//     }
//     // Add missing cities
//     for (int i = 0; i < n; ++i) {
//         if (!visited[i]) fixed_route.push_back(i);
//     }
//     return fixed_route;
// }

// // Main solve function using CUDA
// TSPResult solve(const vector<pair<double, double>>& coordinates) {
//     int n = coordinates.size();
//     int pop_size = 100, generations = 300;
//     double mutation_rate = 0.01;

//     // Convert coordinates to double2
//     vector<double2> host_coordinates(n);
//     for (int i = 0; i < n; ++i) {
//         host_coordinates[i] = {coordinates[i].first, coordinates[i].second};
//     }

//     // Allocate device memory
//     double2* d_coordinates;
//     double* d_distances;
//     int* d_population;
//     double* d_fitness;

//     cudaMalloc(&d_coordinates, n * sizeof(double2));
//     cudaMalloc(&d_distances, n * n * sizeof(double));
//     cudaMalloc(&d_population, pop_size * n * sizeof(int));
//     cudaMalloc(&d_fitness, pop_size * sizeof(double));

//     // Copy coordinates to device
//     cudaMemcpy(d_coordinates, host_coordinates.data(), n * sizeof(double2), cudaMemcpyHostToDevice);

//     // Initialize distance matrix
//     initDistanceMatrix<<<(n * n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_coordinates, d_distances, n);
//     cudaDeviceSynchronize();

//     // Debug: Print distance matrix
//     vector<double> host_distances(n * n);
//     cudaMemcpy(host_distances.data(), d_distances, n * n * sizeof(double), cudaMemcpyDeviceToHost);

//     cout << "Distance matrix:" << endl;
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             cout << host_distances[i * n + j] << " ";
//         }
//         cout << endl;
//     }

//     // Generate initial population on the host
//     auto host_population = generateInitialPopulation(pop_size, n);

//     // Debug: Print generated host population
//     cout << "Generated host population:" << endl;
//     for (const auto& individual : host_population) {
//         for (int gene : individual) {
//             cout << gene << " ";
//         }
//         cout << endl;
//     }

//     // Flatten the population for memory copy
//     vector<int> flat_population;
//     for (const auto& individual : host_population) {
//         flat_population.insert(flat_population.end(), individual.begin(), individual.end());
//     }

//     // Copy population to device
//     cudaMemcpy(d_population, flat_population.data(), pop_size * n * sizeof(int), cudaMemcpyHostToDevice);

//     // Debug: Validate copied population
//     vector<int> host_population_flat(pop_size * n);
//     cudaMemcpy(host_population_flat.data(), d_population, pop_size * n * sizeof(int), cudaMemcpyDeviceToHost);

//     cout << "Copied population to device:" << endl;
//     for (int i = 0; i < pop_size; ++i) {
//         for (int j = 0; j < n; ++j) {
//             cout << host_population_flat[i * n + j] << " ";
//         }
//         cout << endl;
//     }

//     // Evolution loop
//     vector<int> best_route;
//     double best_distance = numeric_limits<double>::infinity();

//     // for (int gen = 0; gen < generations; ++gen) {
//     //     // Calculate fitness
//     //     calculateFitness<<<(pop_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_distances, d_fitness, n, pop_size);
//     //     cudaDeviceSynchronize();

//     //     // Debug: Print fitness values
//     //     vector<double> host_fitness(pop_size);
//     //     cudaMemcpy(host_fitness.data(), d_fitness, pop_size * sizeof(double), cudaMemcpyDeviceToHost);

//     //     cout << "Generation " << gen << " fitness: ";
//     //     for (double f : host_fitness) {
//     //         cout << f << " ";
//     //     }
//     //     cout << endl;

//     //     // Sort population by fitness (using Thrust)
//     //     thrust::device_ptr<double> fitness_ptr(d_fitness);
//     //     thrust::device_ptr<int> population_ptr(d_population);
//     //     thrust::sort_by_key(fitness_ptr, fitness_ptr + pop_size, population_ptr);

//     //     // Mutate population
//     //     mutatePopulation<<<(pop_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, n, mutation_rate, gen);
//     //     cudaDeviceSynchronize();

//     //     // Copy best individual back to host
//     //     vector<int> current_best_route(n);
//     //     cudaMemcpy(current_best_route.data(), d_population, n * sizeof(int), cudaMemcpyDeviceToHost);

//     //     current_best_route = fixRoute(current_best_route, n);

//     //     double current_best_distance;
//     //     cudaMemcpy(&current_best_distance, d_fitness, sizeof(double), cudaMemcpyDeviceToHost);

//     //     if (1.0 / current_best_distance < best_distance) {
//     //         best_distance = 1.0 / current_best_distance;
//     //         best_route = current_best_route;
//     //     }
//     // }


//     int elite_count = pop_size / 10; // Preserve the top 10% of the population

//     for (int gen = 0; gen < generations; ++gen) {
//         // Calculate fitness
//         calculateFitness<<<(pop_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_distances, d_fitness, n, pop_size);
//         cudaDeviceSynchronize();

//         // Sort population by fitness
//         thrust::device_ptr<double> fitness_ptr(d_fitness);
//         thrust::device_ptr<int> population_ptr(d_population);
//         thrust::sort_by_key(fitness_ptr, fitness_ptr + pop_size, population_ptr);

//         // Debug: Verify sorted fitness values
//         vector<double> host_fitness(pop_size);
//         cudaMemcpy(host_fitness.data(), d_fitness, pop_size * sizeof(double), cudaMemcpyDeviceToHost);

//         cout << "Generation " << gen << " sorted fitness: ";
//         for (int i = 0; i < pop_size; ++i) {
//             cout << host_fitness[i] << " ";
//         }
//         cout << endl;

//         // Copy top elites
//         int* d_elite_population;
//         cudaMalloc(&d_elite_population, elite_count * n * sizeof(int));
//         cudaMemcpy(d_elite_population, d_population, elite_count * n * sizeof(int), cudaMemcpyDeviceToDevice);

//         // Debug: Print elite individuals
//         vector<int> host_elite(elite_count * n);
//         cudaMemcpy(host_elite.data(), d_elite_population, elite_count * n * sizeof(int), cudaMemcpyDeviceToHost);

//         cout << "Elite individuals:" << endl;
//         for (int i = 0; i < elite_count; ++i) {
//             for (int j = 0; j < n; ++j) {
//                 cout << host_elite[i * n + j] << " ";
//             }
//             cout << endl;
//         }

//         // Mutate only the non-elite portion of the population
//         mutatePopulation<<<(pop_size - elite_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
//             d_population + elite_count * n, n, mutation_rate, gen);
//         cudaDeviceSynchronize();

//         // Preserve elites in the population
//         cudaMemcpy(d_population, d_elite_population, elite_count * n * sizeof(int), cudaMemcpyDeviceToDevice);
//         cudaFree(d_elite_population);

//         // Copy best individual back to host for tracking
//         vector<int> current_best_route(n);
//         cudaMemcpy(current_best_route.data(), d_population, n * sizeof(int), cudaMemcpyDeviceToHost);

//         double current_best_distance;
//         cudaMemcpy(&current_best_distance, d_fitness, sizeof(double), cudaMemcpyDeviceToHost);

//         if (1.0 / current_best_distance < best_distance) {
//             best_distance = 1.0 / current_best_distance;
//             best_route = current_best_route;
//         }
//     }



//     // Free device memory
//     cudaFree(d_coordinates);
//     cudaFree(d_distances);
//     cudaFree(d_population);
//     cudaFree(d_fitness);

//     // Return the best result
//     TSPResult result;
//     result.cost = best_distance;
//     result.path = best_route;
//     return result;
// }
















// #include <cuda_runtime.h>
// #include <curand_kernel.h>
// #include <thrust/device_vector.h>
// #include <thrust/sort.h>
// #include <vector>
// #include <iostream>
// #include <numeric>
// #include <random>
// #include "../common/algorithms.hpp" // Include the common header for TSPResult

// using namespace std;

// #define BLOCK_SIZE 256

// // Device function to calculate the Euclidean distance between two points
// __device__ double distance(const double2& p1, const double2& p2) {
//     return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
// }

// // Kernel to initialize the distance matrix
// __global__ void initDistanceMatrix(const double2* coordinates, double* distances, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n * n) {
//         int i = idx / n;
//         int j = idx % n;
//         if (i < n && j < n) { // Safe bounds
//             distances[idx] = distance(coordinates[i], coordinates[j]);
//         }
//     }
// }

// // Kernel to calculate fitness values for a population
// __global__ void calculateFitness(const int* population, const double* distances, double* fitness, int n, int pop_size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < pop_size) {
//         const int* route = &population[idx * n];
//         double route_distance = 0.0;

//         for (int i = 0; i < n - 1; ++i) {
//             int from = route[i];
//             int to = route[i + 1];
//             if (from >= 0 && from < n && to >= 0 && to < n) {
//                 route_distance += distances[from * n + to];
//             }
//         }
//         int last = route[n - 1];
//         int first = route[0];
//         if (last >= 0 && last < n && first >= 0 && first < n) {
//             route_distance += distances[last * n + first];
//         }
//         fitness[idx] = (route_distance > 0) ? 1.0 / route_distance : 0.0;
//     }
// }

// // Kernel for mutation
// __global__ void mutatePopulation(int* population, int n, double mutation_rate, int seed) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     curandState state;
//     curand_init(seed + idx, 0, 0, &state);

//     if (idx < n) {
//         int* route = &population[idx * n];
//         for (int i = 0; i < n; ++i) {
//             if (curand_uniform(&state) < mutation_rate) {
//                 int j = curand(&state) % n;
//                 // Swap cities
//                 int temp = route[i];
//                 route[i] = route[j];
//                 route[j] = temp;
//             }
//         }
//     }
// }

// // Kernel to fix routes with duplicates
// // __global__ void fixRouteKernel(int* population, int n, int pop_size) {
// //     for (int idx = 0; idx < pop_size; ++idx) {
// //         // Get the start of the current route in the population
// //         int* route = &population[idx * n];
// //         bool visited[n] = {false}; // Track visited nodes
// //         int fixed_route[n];
// //         int pos = 0;

// //         // Mark visited nodes
// //         for (int i = 0; i < n; ++i) {
// //             if (route[i] >= 0 && route[i] < n) {
// //                 visited[route[i]] = true;
// //             }
// //         }

// //         // Create fixed_route by appending unique nodes from the route
// //         for (int i = 0; i < n; ++i) {
// //             if (route[i] >= 0 && route[i] < n &&
// //                 std::find(fixed_route.begin(), fixed_route.end(), route[i]) == fixed_route.end()) {
// //                 fixed_route.push_back(route[i]);
// //             }
// //         }

// //         // Add missing nodes to complete the route
// //         for (int i = 0; i < n; ++i) {
// //             if (!visited[i]) {
// //                 fixed_route.push_back(i);
// //             }
// //         }

// //         // Copy the fixed route back to the original array
// //         for (int i = 0; i < n; ++i) {
// //             route[i] = fixed_route[i];
// //         }
// //     }
// // }

// __global__ void fixRouteKernel(int* population, int n, int pop_size) {
//     for (int idx = 0; idx < pop_size; ++idx) {
//         int* route = &population[idx * n];
//         bool* visited = new bool[n]; // Replace with n if n is a compile-time constant
//         int* fixed_route = new int[n]; // Replace with n if n is a compile-time constant
//         int pos = 0;

//         for (int i = 0; i < n; ++i) {
//             visited[i] = false;
//         }

//         for (int i = 0; i < n; ++i) {
//             if (route[i] >= 0 && route[i] < n) {
//                 fixed_route[pos++] = route[i];
//                 visited[route[i]] = true; // Mark as visited
//             }
//         }

//         for (int i = 0; i < n; ++i) {
//             if (route[i] >= 0 && route[i] < n &&
//                 [&fixed_route, pos, route, i]() {
//                     for (int k = 0; k < pos; ++k) {
//                         if (fixed_route[k] == route[i]) return false;
//                     }
//                     return true;
//                 }()) {
//                 fixed_route[pos++] = route[i];
//             }
//         }

//         for (int i = 0; i < n; ++i) {
//             if (!visited[i]) {
//                 fixed_route[pos++] = i;
//             }
//         }

//         for (int i = 0; i < n; ++i) {
//             route[i] = fixed_route[i];
//         }
//     }

//     // int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     // if (idx < pop_size) {
//     //     // Get the start of the current route in the population
//     //     int* route = &population[idx * n];
//     //     bool visited[n] = {false}; // Replace with n if n is a compile-time constant
//     //     int fixed_route[n]; // Replace with n if n is a compile-time constant
//     //     int pos = 0;

//     //     // Create fixed_route by appending unique nodes from the route
//     //     for (int i = 0; i < n; ++i) {
//     //         if (route[i] >= 0 && route[i] < n && !visited[route[i]]) {
//     //             fixed_route[pos++] = route[i];
//     //             visited[route[i]] = true; // Mark as visited
//     //         }
//     //     }

//     //     // Add missing nodes to complete the route
//     //     for (int i = 0; i < n; ++i) {
//     //         if (!visited[i]) {
//     //             fixed_route[pos++] = i;
//     //         }
//     //     }

//     //     // Copy the fixed route back to the original array
//     //     for (int i = 0; i < n; ++i) {
//     //         route[i] = fixed_route[i];
//     //     }
//     // }
// }



// // Function to generate random initial population on the host
// vector<vector<int>> generateInitialPopulation(int pop_size, int n) {
//     vector<vector<int>> population(pop_size, vector<int>(n));
//     random_device rd;
//     mt19937 g(rd());

//     for (int i = 0; i < pop_size; ++i) {
//         iota(population[i].begin(), population[i].end(), 0);
//         shuffle(population[i].begin(), population[i].end(), g);
//     }
//     return population;
// }

// // Main solve function using CUDA
// TSPResult solve(const vector<pair<double, double>>& coordinates) {
//     int n = coordinates.size();
//     int pop_size = 100, generations = 300;
//     double mutation_rate = 0.01;

//     // Convert coordinates to double2
//     vector<double2> host_coordinates(n);
//     for (int i = 0; i < n; ++i) {
//         host_coordinates[i] = {coordinates[i].first, coordinates[i].second};
//     }

//     // Allocate device memory
//     double2* d_coordinates;
//     double* d_distances;
//     int* d_population;
//     double* d_fitness;

//     cudaMalloc(&d_coordinates, n * sizeof(double2));
//     cudaMalloc(&d_distances, n * n * sizeof(double));
//     cudaMalloc(&d_population, pop_size * n * sizeof(int));
//     cudaMalloc(&d_fitness, pop_size * sizeof(double));

//     // Copy coordinates to device
//     cudaMemcpy(d_coordinates, host_coordinates.data(), n * sizeof(double2), cudaMemcpyHostToDevice);

//     // Initialize distance matrix
//     initDistanceMatrix<<<(n * n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_coordinates, d_distances, n);
//     cudaDeviceSynchronize();

//     // Generate initial population on the host
//     auto host_population = generateInitialPopulation(pop_size, n);

//     // Flatten the population for memory copy
//     vector<int> flat_population;
//     for (const auto& individual : host_population) {
//         flat_population.insert(flat_population.end(), individual.begin(), individual.end());
//     }

//     // Copy population to device
//     cudaMemcpy(d_population, flat_population.data(), pop_size * n * sizeof(int), cudaMemcpyHostToDevice);

//     int elite_count = pop_size / 10; // Preserve the top 10% of the population

//     vector<int> best_route;
//     double best_distance = numeric_limits<double>::infinity();

//     for (int gen = 0; gen < generations; ++gen) {
//         // Calculate fitness
//         calculateFitness<<<(pop_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_distances, d_fitness, n, pop_size);
//         cudaDeviceSynchronize();

//         // Sort population by fitness
//         thrust::device_ptr<double> fitness_ptr(d_fitness);
//         thrust::device_ptr<int> population_ptr(d_population);
//         thrust::sort_by_key(fitness_ptr, fitness_ptr + pop_size, population_ptr);

//         // Copy top elites
//         int* d_elite_population;
//         cudaMalloc(&d_elite_population, elite_count * n * sizeof(int));
//         cudaMemcpy(d_elite_population, d_population, elite_count * n * sizeof(int), cudaMemcpyDeviceToDevice);

//         // Mutate the rest of the population
//         mutatePopulation<<<(pop_size - elite_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
//             d_population + elite_count * n, n, mutation_rate, gen);
//         cudaDeviceSynchronize();

//         // Fix routes for the entire population
//         fixRouteKernel<<<(pop_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, n, pop_size);
//         cudaDeviceSynchronize();

//         // Preserve elites in the population
//         cudaMemcpy(d_population, d_elite_population, elite_count * n * sizeof(int), cudaMemcpyDeviceToDevice);
//         cudaFree(d_elite_population);

//         // Track the best route
//         vector<int> current_best_route(n);
//         cudaMemcpy(current_best_route.data(), d_population, n * sizeof(int), cudaMemcpyDeviceToHost);

//         double current_best_distance;
//         cudaMemcpy(&current_best_distance, d_fitness, sizeof(double), cudaMemcpyDeviceToHost);

//         if (1.0 / current_best_distance < best_distance) {
//             best_distance = 1.0 / current_best_distance;
//             best_route = current_best_route;
//         }
//     }


//     // Free device memory
//     cudaFree(d_coordinates);
//     cudaFree(d_distances);
//     cudaFree(d_population);
//     cudaFree(d_fitness);

//     // Return the best result
//     TSPResult result;
//     result.cost = best_distance;
//     result.path = best_route;
//     return result;
// }
