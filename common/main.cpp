// #ifdef MPI
#include <mpi.h>
// #endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility> // For std::pair
#include <cstring>
#include "algorithms.hpp" // Include header for algorithm-specific init and solve functions

void printTSPResult(const TSPResult &result, std::ostream &out) {
    out << std::fixed << std::setprecision(6); // Set fixed-point notation with 6 decimal places
    out << "Cost: " << result.cost << std::endl;
    out << "Path: ";
    for (size_t i = 0; i < result.path.size(); ++i) {
        out << result.path[i];
        if (i < result.path.size() - 1) {
            out << " -> ";
        }
    }
    out << std::endl;
}

// Function to get the executable name and construct the output file path
std::string getOutputFilePath(const char *argv0) {
    // Extract the executable name
    std::string exec_name = std::filesystem::path(argv0).stem();
    return "build/" + exec_name + ".out"; // Create the output file path
}

// Function to read the CSV and store it as a vector of pairs (x, y coordinates)
void read_csv(const std::string &file_path, std::vector<std::pair<double, double>> &coordinates) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream line_stream(line);
        std::string x_str, y_str;

        if (std::getline(line_stream, x_str, ',') && std::getline(line_stream, y_str, ',')) {
            double x = std::stod(x_str);
            double y = std::stod(y_str);
            coordinates.emplace_back(x, y); // Add the coordinates as a pair
        }
    }

    file.close();
}

int main(int argc, char **argv) {
// #ifdef MPI
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
// #endif

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " --csv <file_path>" << std::endl;
        return 1;
    }

    std::string csv_file;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            if (i + 1 < argc) {
                csv_file = argv[++i];
            } else {
                std::cerr << "Error: Missing value for --csv" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            return 1;
        }
    }

    // Read the CSV file
    std::vector<std::pair<double, double>> coordinates;
    read_csv(csv_file, coordinates);

    // Solve and capture the result
// #ifdef MPI
    TSPResult result = solve(coordinates, rank, num_procs);
    MPI_Finalize();
// #else
    // TSPResult result = solve(coordinates);
// #endif

    // Determine the output file path
    std::string output_file_path = getOutputFilePath(argv[0]);

    // Open the output file
    std::ofstream out_file(output_file_path);
    if (!out_file.is_open()) {
        std::cerr << "Error: Unable to open output file: " << output_file_path << std::endl;
        return 1;
    }

    // Print the result to the file and to the terminal
    printTSPResult(result, out_file);
    printTSPResult(result, std::cout);

    out_file.close();
    

    std::cout << "Result written to " << output_file_path << std::endl;

    return 0;
}