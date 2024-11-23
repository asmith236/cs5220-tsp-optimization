#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <limits> // For numeric_limits

// Constants
constexpr int n = 5; // Number of nodes in the graph
constexpr int MAX = std::numeric_limits<int>::max(); // Use a large value as infinity

// Distance matrix representing the graph
extern int dist[n][n];

#endif // CONSTANTS_H
