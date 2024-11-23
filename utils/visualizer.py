import sys
import numpy as np
import matplotlib.pyplot as plt

def read_csv_coordinates(size):
    """Read coordinates from CSV file based on size"""
    filename = f"data/{size}.csv"
    try:
        coordinates = np.loadtxt(filename, delimiter=',')
        return coordinates
    except Exception as e:
        print(f"Error reading coordinates from {filename}: {e}")
        sys.exit(1)

def read_solution(outfile):
    """Read solution path from .out file"""
    try:
        with open(outfile, 'r') as f:
            # Remove parentheses and split by comma
            path_str = f.readline().strip('()\n')
            path = list(map(int, path_str.split(',')))
        return path
    except Exception as e:
        print(f"Error reading solution from {outfile}: {e}")
        sys.exit(1)

def visualize_path(coordinates, path):
    """Create visualization of TSP path"""
    plt.figure(figsize=(10, 10))
    
    # Plot all points
    plt.plot(coordinates[:, 0], coordinates[:, 1], 'ro', markersize=8)
    
    # Plot path
    for i in range(len(path)-1):
        start = path[i]
        end = path[i+1]
        plt.plot([coordinates[start][0], coordinates[end][0]], 
                [coordinates[start][1], coordinates[end][1]], 
                'b-', linewidth=1)
    
    # Connect last point back to first
    plt.plot([coordinates[path[-1]][0], coordinates[path[0]][0]], 
            [coordinates[path[-1]][1], coordinates[path[0]][1]], 
            'b-', linewidth=1)
    
    # Add city numbers (using path order)
    for i, idx in enumerate(path):
        x, y = coordinates[idx]
        plt.annotate(f'{i+1}', 
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold')
    
    plt.title('TSP Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.margins(0.2)
    
    plt.savefig('tsp_path.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python visualizer.py <size> <solution.out>")
        sys.exit(1)
    
    size = sys.argv[1]
    outfile = sys.argv[2]
    
    coordinates = read_csv_coordinates(size)
    path = read_solution(outfile)
    visualize_path(coordinates, path)