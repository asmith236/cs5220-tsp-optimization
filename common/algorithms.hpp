#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP

#include <vector>
#include <utility> // For std::pair
#include <bits/stdc++.h>
using namespace std;

struct TSPResult {
    int cost;
    vector<int> path;
};

TSPResult solve(const std::vector<std::pair<double, double>> &coordinates);

double distance(const std::pair<double, double>& p1, const std::pair<double, double>& p2);

#endif // ALGORITHMS_HPP