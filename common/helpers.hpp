#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <vector>
#include <string>
#include <utility>

// Function declarations
void loadMatrixFromCSV(const std::string& filename);
void addExtraRowAndColumn(std::vector<std::vector<int>>& matrix);
void printMatrix(std::vector<std::vector<int>>& matrix);
double distance(const std::pair<double, double>& p1, const std::pair<double, double>& p2);

#endif
