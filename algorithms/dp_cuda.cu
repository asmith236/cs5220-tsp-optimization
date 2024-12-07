// #include <bits/stdc++.h>
// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <algorithm> // For std::next_permutation
// #include <limits>    // For std::numeric_limits
// #include "../common/algorithms.hpp" // Include the common constants file

// using namespace std;

// #define MAX INT_MAX// Large value as infinity

// // Function to compute distance
// __host__ __device__
// double compute_distance(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
//     return std::sqrt((p1.first - p2.first) * (p1.first - p2.first) + 
//                      (p1.second - p2.second) * (p1.second - p2.second));
// }

// __global__
// void tsp_kernel(double* dp, int* parent, const double* distances, int n, int subset_size) {
//     __shared__ double shared_distances[32][32]; // Shared memory for distances
//     int mask = blockIdx.x;                     // Each block handles one subset
//     int u = threadIdx.x;                       // Each thread handles one node

//     if (mask >= (1 << n) || __popc(mask) != subset_size || u >= n) return;

//     // Load distances into shared memory
//     if (u < n) {
//         for (int v = 0; v < n; v++) {
//             shared_distances[u][v] = distances[u * n + v];
//         }
//     }
//     __syncthreads();

//     if (!(mask & (1 << u))) return;

//     int prev_mask = mask ^ (1 << u);
//     double min_cost = MAX;
//     int best_parent = -1;

//     for (int v = 0; v < n; v++) {
//         if (!(prev_mask & (1 << v))) continue;
//         double new_cost = dp[prev_mask * n + v] + shared_distances[v][u];
//         if (new_cost < min_cost) {
//             min_cost = new_cost;
//             best_parent = v;
//         }
//     }

//     dp[mask * n + u] = min_cost;
//     parent[mask * n + u] = best_parent;
// }



// // Host function to solve TSP
// TSPResult solve(const std::vector<std::pair<double, double>>& coordinates) {
//     int n = coordinates.size();
//     int full_mask = (1 << n) - 1;

//     // Precompute distances
//     std::vector<double> distances(n * n, 0.0);
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             distances[i * n + j] = compute_distance(coordinates[i], coordinates[j]);
//         }
//     }

//     // Allocate DP table and parent table
//     size_t dp_size = (1 << n) * n * sizeof(double);
//     size_t parent_size = (1 << n) * n * sizeof(int);
//     std::vector<double> dp_host((1 << n) * n, MAX);
//     std::vector<int> parent_host((1 << n) * n, -1);
//     dp_host[(1 << 0) * n + 0] = 0; // Equivalent to dp[1][0] = 0 in Python

//     double* dp_device;
//     int* parent_device;
//     double* distances_device;
//     cudaMalloc(&dp_device, dp_size);
//     cudaMalloc(&parent_device, parent_size);
//     cudaMalloc(&distances_device, n * n * sizeof(double));

//     cudaMemcpy(dp_device, dp_host.data(), dp_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(parent_device, parent_host.data(), parent_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(distances_device, distances.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

//     // Launch kernel for each subset size
//     for (int subset_size = 2; subset_size <= n; subset_size++) {
//         int num_blocks = (1 << n);  // One block for each subset
//         int num_threads = n;       // One thread for each node
//         tsp_kernel<<<num_blocks, num_threads>>>(dp_device, parent_device, distances_device, n, subset_size);
//         cudaDeviceSynchronize();
//     }

//     // Copy DP table and parent table back to host
//     cudaMemcpy(dp_host.data(), dp_device, dp_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(parent_host.data(), parent_device, parent_size, cudaMemcpyDeviceToHost);

//     // printf("DP:\n");
//     // for (int mask = 0; mask < (1 << n); mask++) {  // Iterate through all subsets (2^n subsets)
//     //     for (int u = 0; u < n; u++) {  // Iterate through each node in the subset
//     //         printf("%f ", dp_host[mask * n + u]);
//     //     }
//     //     printf("\n");
//     // }

//     // Find optimal cost and last node
//     double opt = MAX;
//     int last_node = -1;
//     for (int k = 1; k < n; k++) {
//         double new_cost = dp_host[full_mask * n + k] + distances[k * n];
//         if (new_cost < opt) {
//             opt = new_cost;
//             last_node = k;
//         }
//     }

//     // Reconstruct the path
//     std::vector<int> path;
//     int current_mask = full_mask;
//     int current_node = last_node;
//     while (current_node != -1) {
//         path.push_back(current_node);
//         int temp = current_node;
//         current_node = parent_host[current_mask * n + current_node];
//         current_mask ^= (1 << temp);
//     }
//     // path.push_back(0); // Add the starting node
//     std::reverse(path.begin(), path.end());

//     // Free GPU memory
//     cudaFree(dp_device);
//     cudaFree(parent_device);
//     cudaFree(distances_device);

//     // Return the result
//     return TSPResult{opt, path};
// }


#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include "../common/algorithms.hpp" // Include the common constants file

constexpr double MAX = std::numeric_limits<double>::max();

// Function to compute distance
__host__ __device__
double compute_distance(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    return std::sqrt((p1.first - p2.first) * (p1.first - p2.first) + 
                     (p1.second - p2.second) * (p1.second - p2.second));
}

__global__
void tsp_kernel(double* dp, int* parent, const double* distances, int n, int subset_size) {
    __shared__ double shared_distances[32][32]; // Shared memory for distances
    int mask = blockIdx.x;                     // Each block handles one subset
    int u = threadIdx.x;                       // Each thread handles one node

    if (mask >= (1 << n) || __popc(mask) != subset_size || u >= n) return;

    // Load distances into shared memory
    if (u < n) {
        for (int v = 0; v < n; v++) {
            shared_distances[u][v] = distances[u * n + v];
        }
    }
    __syncthreads();

    if (!(mask & (1 << u))) return;

    int prev_mask = mask ^ (1 << u);
    double min_cost = MAX;
    int best_parent = -1;

    for (int v = 0; v < n; v++) {
        if (!(prev_mask & (1 << v))) continue;
        double new_cost = dp[prev_mask * n + v] + shared_distances[v][u];
        if (new_cost < min_cost) {
            min_cost = new_cost;
            best_parent = v;
        }
    }

    dp[mask * n + u] = min_cost;
    parent[mask * n + u] = best_parent;
}

TSPResult solve(const std::vector<std::pair<double, double>>& coordinates, int rank, int num_procs) {
    int n = coordinates.size();
    size_t full_mask = (1ULL << n) - 1;

    // Precompute distances
    std::vector<double> distances(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            distances[i * n + j] = compute_distance(coordinates[i], coordinates[j]);
        }
    }

    // Allocate local DP and parent tables
    size_t dp_size = (1ULL << n) * n / num_procs;  // Number of elements per process
    size_t parent_size = dp_size;
    std::vector<double> dp_local(dp_size, MAX);
    std::vector<int> parent_local(parent_size, -1);

    // Initialize base case
    if (rank == 0) dp_local[1 * n] = 0;  // Equivalent to dp[1][0] = 0

    double* dp_device;
    int* parent_device;
    double* distances_device;
    cudaMalloc(&dp_device, dp_size * sizeof(double));
    cudaMalloc(&parent_device, parent_size * sizeof(int));
    cudaMalloc(&distances_device, n * n * sizeof(double));

    cudaMemcpy(distances_device, distances.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    for (int subset_size = 2; subset_size <= n; subset_size++) {
        int num_blocks = 1 << n;  // One block for each subset
        tsp_kernel<<<num_blocks, n>>>(dp_device, parent_device, distances_device, n, subset_size);
        cudaDeviceSynchronize();

        // Copy results back to host for synchronization
        cudaMemcpy(dp_local.data(), dp_device, dp_size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(parent_local.data(), parent_device, parent_size * sizeof(int), cudaMemcpyDeviceToHost);

        // Synchronize results across MPI processes
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, dp_local.data(), dp_size, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent_local.data(), parent_size, MPI_INT, MPI_COMM_WORLD);
    }

    // Gather full DP table on rank 0
    std::vector<double> dp_full;
    std::vector<int> parent_full;
    if (rank == 0) {
        dp_full.resize((1ULL << n) * n, MAX);
        parent_full.resize((1ULL << n) * n, -1);
    }

    MPI_Gather(dp_local.data(), dp_size, MPI_DOUBLE, dp_full.data(), dp_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(parent_local.data(), parent_size, MPI_INT, parent_full.data(), parent_size, MPI_INT, 0, MPI_COMM_WORLD);

    TSPResult result;

    if (rank == 0) {
        // Find optimal cost and last node
        double opt = MAX;
        int last_node = -1;
        for (int k = 1; k < n; k++) {
            double new_cost = dp_full[full_mask * n + k] + distances[k * n];
            if (new_cost < opt) {
                opt = new_cost;
                last_node = k;
            }
        }

        // Reconstruct path
        std::vector<int> path;
        int current_mask = full_mask;
        int current_node = last_node;
        while (current_node != -1) {
            path.push_back(current_node);
            int temp = current_node;
            current_node = parent_full[current_mask * n + current_node];
            current_mask ^= (1 << temp);
        }
        std::reverse(path.begin(), path.end());

        result = TSPResult{opt, path};

        std::cout << "Optimal cost: " << opt << std::endl;
        std::cout << "Optimal path: ";
        for (int node : path) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(dp_device);
    cudaFree(parent_device);
    cudaFree(distances_device);

    return result;
}
