#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP
#include <vector>
#include <utility> // For std::pair
#include <bits/stdc++.h>
using namespace std;

struct TSPResult {
    double cost;
    vector<int> path;
};

double distance(const std::pair<double, double>& p1, const std::pair<double, double>& p2);

TSPResult solve(const std::vector<std::pair<double, double>> &coordinates, int rank, int num_procs);
// TSPResult solve(const std::vector<std::pair<double, double>> &coordinates);

#endif // ALGORITHMS_HPP