#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../common/algorithms.hpp"

const int BLOCK_SIZE = 256;

__global__ void findNearestCityKernel(
    const double* coords_x, // Array of x coordinates
    const double* coords_y, // Array of y coordinates
    const char* visited, // Array of visited cities (1 if visited, 0 if not)
    int current_city, // idx of the current city
    int n, // Total number of cities
    double* min_distances, //Output: minimum distance to each city
    int* next_cities // Output: next city to visit
) {
    __shared__ double shared_min_distances[BLOCK_SIZE]; // Shared memory for minimum distances
    __shared__ int shared_next_cities[BLOCK_SIZE]; // Shared memory for next cities
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load current city coordinates into shared memory
    __shared__ double current_coords[2];
    if (tid == 0) { // Only the first thread in the block loads the current city coordinates
        current_coords[0] = coords_x[current_city];
        current_coords[1] = coords_y[current_city];
    }
    __syncthreads(); // Ensures all threads wait until the coordinates are loaded
    
    double local_min = INFINITY;
    int local_next = -1;
    
    // Each thread maintains its own minimum distance and corresponding city
    for (int j = gid; j < n; j += blockDim.x * gridDim.x) { //blockDim.x * gridDim.x is the total number of threads
        if (!visited[j]) { // If the city has not been visited
            double dx = current_coords[0] - coords_x[j];
            double dy = current_coords[1] - coords_y[j];
            double dist = sqrt(dx * dx + dy * dy);
            if (dist < local_min) { // If the distance is less than the local minimum
                local_min = dist;
                local_next = j;
            }
        }
    }

    // For example, if n = 1000, and we have 256 threads, then each thread will have to calculate 4 cities
    
    shared_min_distances[tid] = local_min;
    shared_next_cities[tid] = local_next;
    __syncthreads(); // Ensures all threads wait until the minimum distances and next cities are loaded
    
    // Block that performs parallel reduction to find the global minimum distance and corresponding city

    // Explanation of what is parallel reduction:
    /* In parallel reduction, we are trying to find the minimum distance and the corresponding city from the shared memory
    array shared_min_distances. for example, if we have 256 threads, then we have 256 elements in shared_min_distances.
    We are trying to find the minimum distance and the corresponding city from these 256 elements. 
    the stride initially is 128, then 64, then 32, then 16, then 8, then 4, then 2, then 1.
    so we are comparing the elements at index 0 and 128 (along with 1 and 129, 2 and 130 etc etc), then 0 and 64, then 0 and 32, then 0 and 16, then 0 and 8, then 0 and 4, then 0 and 2, then 0 and 1.
    and we are updating the minimum distance and the corresponding city accordingly.
    */
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_min_distances[tid + stride] < shared_min_distances[tid]) {
                shared_min_distances[tid] = shared_min_distances[tid + stride];
                shared_next_cities[tid] = shared_next_cities[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // After the parallel reduction, the minimum distance and the corresponding city will be at index 0
    if (tid == 0) {
        min_distances[blockIdx.x] = shared_min_distances[0];
        next_cities[blockIdx.x] = shared_next_cities[0];
    }

    /* Benefits of parallel reduction:
    1. Reduces complexity from O(n) to O(log(n))
    2. Reduces the number of global memory accesses
    */
}

TSPResult solve(const std::vector<std::pair<double, double>>& coordinates) {
    int n = coordinates.size();
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem); // Command that gets the available free memory on the GPU
    
    // Pinned memory: memory that is not swapped out to disk optimizes memory transfer between CPU and GPU
    // Putting x and y coordinates into pinned memory
    double *h_coords_x, *h_coords_y;
    cudaMallocHost(&h_coords_x, n * sizeof(double));
    cudaMallocHost(&h_coords_y, n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        h_coords_x[i] = coordinates[i].first;
        h_coords_y[i] = coordinates[i].second;
    }
    
    /* Allocating GPU memory for coordaiantes.
    Copying data from CPU to GPU*/
    double *d_coords_x, *d_coords_y;
    cudaMalloc(&d_coords_x, n * sizeof(double));
    cudaMalloc(&d_coords_y, n * sizeof(double));
    
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
    visited[current_city] = 1;  // 1 stands for true, means that city has been visited.
    cudaMemcpy(d_visited, visited.data(), n * sizeof(char), cudaMemcpyHostToDevice);
    
    // Allocate device memory for minimum distance search
    double* d_min_distances;
    int* d_next_cities;
    cudaMalloc(&d_min_distances, BLOCK_SIZE * sizeof(double)); // We are going to use BLOCK_SIZE threads
    cudaMalloc(&d_next_cities, BLOCK_SIZE * sizeof(int)); 
    
    const int threads_per_block = BLOCK_SIZE;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;
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
    
    // Cleaning up all the variables that we have used
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
