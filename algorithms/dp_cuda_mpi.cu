#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include "../common/algorithms.hpp" // Include the common constants file

#define MAX INT_MAX

// Function to compute distance
__host__ __device__
double compute_distance(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    return std::sqrt((p1.first - p2.first) * (p1.first - p2.first) + 
                     (p1.second - p2.second) * (p1.second - p2.second));
}

__global__
void tsp_kernel(double* dp, int* parent, const double* distances, int n, int subset_size, int start_mask, int end_mask) {
    int mask = start_mask + blockIdx.x; // Each block handles one subset
    int u = threadIdx.x;               // Each thread handles one node

    if (mask >= end_mask || __popc(mask) != subset_size || u >= n) return;

    if (!(mask & (1 << u))) return;

    int prev_mask = mask ^ (1 << u);
    double min_cost = MAX;
    int best_parent = -1;

    for (int v = 0; v < n; v++) {
        if (!(prev_mask & (1 << v))) continue;
        double new_cost = dp[prev_mask * n + v] + distances[v * n + u];
        if (new_cost < min_cost) {
            min_cost = new_cost;
            best_parent = v;
        }
    }

    dp[mask * n + u] = min_cost;
    parent[mask * n + u] = best_parent;

    // Debugging output for kernel
    if (u == 0 && mask == start_mask) {
        printf("Kernel debug: mask=%d, subset_size=%d, u=%d, min_cost=%f\n", mask, subset_size, u, min_cost);
    }
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

    // Debugging distances
    // if (rank == 0) {
    //     std::cout << "Distances:" << std::endl;
    //     for (int i = 0; i < n; i++) {
    //         for (int j = 0; j < n; j++) {
    //             std::cout << distances[i * n + j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // Allocate local DP and parent tables
    size_t dp_size = ((1ULL << n) * n) / num_procs;  // Number of elements, not bytes
    size_t parent_size = ((1ULL << n) * n) / num_procs; // Number of elements, not bytes
    std::vector<double> dp_local(dp_size, MAX);
    std::vector<int> parent_local(parent_size, -1);

    double* dp_device;
    int* parent_device;
    double* distances_device;
    cudaMalloc(&dp_device, dp_size * sizeof(double));
    cudaMalloc(&parent_device, parent_size * sizeof(int));
    cudaMalloc(&distances_device, n * n * sizeof(double));

    cudaMemcpy(distances_device, distances.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    int local_mask_start = rank * (1ULL << (n - __builtin_ctz(num_procs)));
    int local_mask_end = (rank + 1) * (1ULL << (n - __builtin_ctz(num_procs)));

    for (int subset_size = 2; subset_size <= n; subset_size++) {
        int num_blocks = local_mask_end - local_mask_start;
        tsp_kernel<<<num_blocks, n>>>(dp_device, parent_device, distances_device, n, subset_size, local_mask_start, local_mask_end);
        cudaDeviceSynchronize();

        // Debugging DP table after kernel execution
        cudaMemcpy(dp_local.data(), dp_device, dp_size * sizeof(double), cudaMemcpyDeviceToHost);
        // if (rank == 0) {
        //     std::cout << "DP Table after subset_size=" << subset_size << ":\n";
        //     for (size_t i = 0; i < dp_size; i++) {
        //         std::cout << dp_local[i] << " ";
        //         if ((i + 1) % n == 0) std::cout << std::endl;
        //     }
        // }

        // Gather results from all GPUs
        std::vector<double> dp_gather(dp_size * num_procs, MAX);
        MPI_Allgather(dp_local.data(), dp_size, MPI_DOUBLE, dp_gather.data(), dp_size, MPI_DOUBLE, MPI_COMM_WORLD);
        dp_local = dp_gather;

        std::vector<int> parent_gather(parent_size * num_procs, -1);
        MPI_Allgather(parent_local.data(), parent_size, MPI_INT, parent_gather.data(), parent_size, MPI_INT, MPI_COMM_WORLD);
        parent_local = parent_gather;
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
        // Debugging DP full table
        // std::cout << "Full DP Table:" << std::endl;
        // for (size_t i = 0; i < dp_full.size(); i++) {
        //     std::cout << dp_full[i] << " ";
        //     if ((i + 1) % n == 0) std::cout << std::endl;
        // }

        // Find optimal cost and path on rank 0
        double opt = MAX;
        int last_node = -1;
        for (int k = 1; k < n; k++) {
            double new_cost = dp_full[full_mask * n + k] + distances[k * n];
            if (new_cost < opt) {
                opt = new_cost;
                last_node = k;
            }
        }
        std::cout << "Optimal cost found: " << opt << std::endl;

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

        // Debugging reconstructed path
        std::cout << "Reconstructed Path: ";
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
