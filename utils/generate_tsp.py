# generate_tsp.py
import numpy as np

def generate_tsp_instance(n_cities, min_dist=1, max_dist=100, symmetric=True):
    """Generate a random TSP instance with n cities"""
    if symmetric:
        # Create upper triangle
        distances = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(i+1, n_cities):
                distances[i][j] = np.random.randint(min_dist, max_dist)
                distances[j][i] = distances[i][j]  # Mirror for symmetry
    else:
        distances = np.random.randint(min_dist, max_dist, size=(n_cities, n_cities))
        np.fill_diagonal(distances, 0)  # Set diagonal to 0
    
    return distances

def save_tsp_instance(distances, filename):
    """Save TSP instance to file"""
    n_cities = len(distances)
    with open(filename, 'w') as f:
        f.write(f"{n_cities}\n")  # First line: number of cities
        for row in distances:
            f.write(" ".join(map(str, row.astype(int))) + "\n")

# Generate test cases for sizes 4 to 10
for size in range(4, 11):
    distances = generate_tsp_instance(size)
    save_tsp_instance(distances, f"test_case_{size}.txt")