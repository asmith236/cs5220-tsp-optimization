import numpy as np
import os

def generate_tsp_coordinates(n_cities, min_coord=-3.0, max_coord=3.0):
    """Generate n_cities random coordinates with scientific notation"""
    coords = np.random.uniform(min_coord, max_coord, size=(n_cities, 2))
    return coords

def save_coordinates(coords, filename):
    """Save coordinates to CSV with specific format"""
    os.makedirs('data', exist_ok=True)
    # Set numpy print options to match desired scientific notation
    np.set_printoptions(precision=15, suppress=False)
    fmt='%.15e,%.15e'  # Format string for scientific notation with 15 decimal places
    np.savetxt(f'data/{filename}.csv', coords, delimiter=',', fmt=fmt)

# Define dataset sizes
sizes = {
    'huge': 10_000,
    'gigantic': 100_000,
}

# Generate and save each dataset
for name, size in sizes.items():
    print(f"Generating {name} dataset with {size} cities...")
    coords = generate_tsp_coordinates(size)
    save_coordinates(coords, name)
    print(f"Saved {name}.csv")