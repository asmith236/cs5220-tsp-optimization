import argparse
import os

# Define the correct outputs for datasets and algorithms
CORRECT_SOLUTIONS = {
    "tiny": {
        "brute": {"cost": 12.516978, "path": [0, 4, 1, 5, 8, 2, 7, 6, 9, 3]},
        "dp": {"cost": 12.516978, "path": [0, 4, 1, 5, 8, 2, 7, 6, 9, 3]},
        "greedy": {"cost": 12.516978, "path": [0, 4, 1, 5, 8, 2, 7, 6, 9, 3]},
        "genetic": {"cost": 12.516978, "path": [0, 4, 1, 5, 8, 2, 7, 6, 9, 3]},
    },
    # Add entries for small, medium, and large datasets here
}

def parse_out_file(file_path):
    """Parse the .out file to extract the cost and path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract cost
    cost_line = next(line for line in lines if line.startswith("Cost:"))
    cost = float(cost_line.split(":")[1].strip())

    # Extract path
    path_line = next(line for line in lines if line.startswith("Path:"))
    path = [int(x) for x in path_line.split(":")[1].strip().split("->")]

    return cost, path

def is_correct(actual_cost, actual_path, expected_cost, expected_path):
    """Check if the cost and path are correct."""
    # Check cost
    if round(actual_cost, 2) != round(expected_cost, 2):
        return False

    # Check path (including rotations and reversals)
    n = len(expected_path)
    for i in range(n):
        rotated_path = expected_path[i:] + expected_path[:i]
        reversed_path = rotated_path[::-1]
        if actual_path == rotated_path or actual_path == reversed_path:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Verify TSP algorithm correctness.")
    parser.add_argument("--alg", choices=["brute", "dp", "greedy", "genetic"], required=True, help="Algorithm to verify.")
    parser.add_argument("--csv", choices=["tiny", "small", "medium", "large"], required=True, help="Dataset to use for verification.")

    args = parser.parse_args()

    # Get correct solution
    if args.csv not in CORRECT_SOLUTIONS or args.alg not in CORRECT_SOLUTIONS[args.csv]:
        raise ValueError(f"No correct solution found for dataset '{args.csv}' and algorithm '{args.alg}'.")

    correct_solution = CORRECT_SOLUTIONS[args.csv][args.alg]
    expected_cost = correct_solution["cost"]
    expected_path = correct_solution["path"]

    # Parse the .out file
    try:
        actual_cost, actual_path = parse_out_file(f"build/{args.alg}.out")
    except Exception as e:
        print(f"Error reading .out file: {e}")
        return

    # Verify correctness
    if not is_correct(actual_cost, actual_path, expected_cost, expected_path):
        print("Incorrect!")
        print(f"Expected: cost = {expected_cost}, path = {expected_path}")
        print(f"Got: cost = {actual_cost}, path = {actual_path}")

if __name__ == "__main__":
    main()
