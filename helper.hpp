#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;
vector<vector<int>> loadMatrixFromCSV(const string& filename) {
    ifstream file(filename);
    vector<vector<int>> matrix;
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<int> row;

        while (getline(ss, cell, ',')) {
            row.push_back(stoi(cell)); // Convert string to int
        }

        matrix.push_back(row);
    }

    return matrix;
}

void addExtraRowAndColumn(vector<vector<int>>& graph) {
    int originalSize = graph.size();
    
    // Add a new row of zeros at the front
    graph.insert(graph.begin(), vector<int>(originalSize + 1, 0));
    
    // Add a zero at the beginning of each existing row
    for (int i = 1; i <= originalSize; i++) { // Start from 1 to avoid modifying the newly added row
        graph[i].insert(graph[i].begin(), 0);
    }
}

void printMatrix(vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }
}