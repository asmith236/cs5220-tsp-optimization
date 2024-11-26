#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../common/algorithms.hpp"

const int TILE_SIZE = 1024;
const int BLOCK_SIZE = 256;

__global__ void findNearestCityKernel(
    const double* coords_x,
    const double* coords_y,
    const char* visited,
    int current_city,
    int n,
    double* min_distances,
    int* next_cities
) {
    __shared__ double shared_min_distances[BLOCK_SIZE];
    __shared__ int shared_next_cities[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load current city coordinates into shared memory
    __shared__ double current_coords[2];
    if (tid == 0) {
        current_coords[0] = coords_x[current_city];
        current_coords[1] = coords_y[current_city];
    }
    __syncthreads();
    
    double local_min = INFINITY;
    int local_next = -1;
    
    // Process multiple cities per thread
    for (int j = gid; j < n; j += blockDim.x * gridDim.x) {
        if (!visited[j]) {
            double dx = current_coords[0] - coords_x[j];
            double dy = current_coords[1] - coords_y[j];
            double dist = sqrt(dx * dx + dy * dy);
            if (dist < local_min) {
                local_min = dist;
                local_next = j;
            }
        }
    }
    
    shared_min_distances[tid] = local_min;
    shared_next_cities[tid] = local_next;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_min_distances[tid + stride] < shared_min_distances[tid]) {
                shared_min_distances[tid] = shared_min_distances[tid + stride];
                shared_next_cities[tid] = shared_next_cities[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        min_distances[blockIdx.x] = shared_min_distances[0];
        next_cities[blockIdx.x] = shared_next_cities[0];
    }
}

TSPResult solve(const std::vector<std::pair<double, double>>& coordinates) {
    int n = coordinates.size();
    
    // Check available GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // Allocate pinned memory for coordinates
    double *h_coords_x, *h_coords_y;
    cudaMallocHost(&h_coords_x, n * sizeof(double));
    cudaMallocHost(&h_coords_y, n * sizeof(double));
    
    // Copy coordinates to pinned memory
    for (int i = 0; i < n; i++) {
        h_coords_x[i] = coordinates[i].first;
        h_coords_y[i] = coordinates[i].second;
    }
    
    // Allocate device memory for coordinates
    double *d_coords_x, *d_coords_y;
    cudaMalloc(&d_coords_x, n * sizeof(double));
    cudaMalloc(&d_coords_y, n * sizeof(double));
    
    // Copy coordinates to device
    cudaMemcpy(d_coords_x, h_coords_x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_y, h_coords_y, n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Initialize path construction variables
    std::vector<int> path;
    double total_cost = 0.0;
    std::vector<char> visited(n, 0);  // Using char instead of bool
    
    // Allocate device memory for visited array
    char* d_visited;
    cudaMalloc(&d_visited, n * sizeof(char));
    
    // Start from city 0
    int current_city = 0;
    path.push_back(current_city);
    visited[current_city] = 1;  // Using 1 instead of true
    cudaMemcpy(d_visited, visited.data(), n * sizeof(char), cudaMemcpyHostToDevice);
    
    // Allocate device memory for minimum distance search
    double* d_min_distances;
    int* d_next_cities;
    cudaMalloc(&d_min_distances, BLOCK_SIZE * sizeof(double));
    cudaMalloc(&d_next_cities, BLOCK_SIZE * sizeof(int));
    
    // Main loop
    while (path.size() < n) {
        // Launch kernel to find nearest city
        findNearestCityKernel<<<BLOCK_SIZE, BLOCK_SIZE>>>(
            d_coords_x,
            d_coords_y,
            d_visited,
            current_city,
            n,
            d_min_distances,
            d_next_cities
        );
        
        // Copy results back to host
        std::vector<double> h_min_distances(BLOCK_SIZE);
        std::vector<int> h_next_cities(BLOCK_SIZE);
        cudaMemcpy(h_min_distances.data(), d_min_distances, BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_next_cities.data(), d_next_cities, BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Find global minimum
        double min_distance = INFINITY;
        int next_city = -1;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (h_min_distances[i] < min_distance) {
                min_distance = h_min_distances[i];
                next_city = h_next_cities[i];
            }
        }
        
        // Update path
        current_city = next_city;
        path.push_back(current_city);
        visited[current_city] = 1;  // Using 1 instead of true
        cudaMemcpy(d_visited, visited.data(), n * sizeof(char), cudaMemcpyHostToDevice);
        total_cost += min_distance;
    }
    
    // Calculate distance back to start
    double final_distance;
    double dx = h_coords_x[path.back()] - h_coords_x[path[0]];
    double dy = h_coords_y[path.back()] - h_coords_y[path[0]];
    final_distance = sqrt(dx * dx + dy * dy);
    total_cost += final_distance;
    
    // Clean up
    cudaFreeHost(h_coords_x);
    cudaFreeHost(h_coords_y);
    cudaFree(d_coords_x);
    cudaFree(d_coords_y);
    cudaFree(d_visited);
    cudaFree(d_min_distances);
    cudaFree(d_next_cities);
    
    TSPResult result;
    result.cost = total_cost;
    result.path = path;
    return result;
}
