#!/usr/bin/env python3
import subprocess
import time
import argparse
import matplotlib.pyplot as plt

def run_implementation(exec_path: str, dataset: str) -> float:
    """Run implementation with dataset and return execution time"""
    try:
        start = time.time()
        result = subprocess.run([exec_path, '--csv', f'data/{dataset}.csv'], 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        end = time.time()
        return end - start
    except Exception as e:
        print(f"Error running {exec_path} with {dataset}: {e}")
        return -1

def plot_results(results):
    """Plot timing results for all implementations"""
    plt.figure(figsize=(12, 8))
    
    colors = {
        'brute': 'b',
        'greedy': 'g',
        'genetic': 'r',
        'dp': 'purple',
        'greedy_omp': 'y'
    }
    
    datasets = ['tiny', 'small', 'medium', 'large','huge','gigantic']
    x_pos = range(len(datasets))
    
    for impl, timings in results.items():
        if timings:  # Only plot if we have data
            plt.plot(x_pos, [timings[d] for d in datasets], 
                    f'{colors[impl]}-o', 
                    label=f'{impl.capitalize()} Implementation')
    
    plt.xticks(x_pos, datasets)
    plt.xlabel('Dataset Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('TSP Implementation Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visibility
    
    plt.savefig('comparison_results.png')
    print("Plot saved as comparison_results.png")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")

if __name__ == "__main__":
    all_implementations = ['brute', 'greedy', 'genetic', 'dp', 'greedy_omp']
    
    parser = argparse.ArgumentParser(description='Compare TSP implementations')
    parser.add_argument('implementations', 
                       nargs='*',
                       choices=all_implementations,
                       help='Implementations to compare. If none specified, runs all.')
    
    args = parser.parse_args()
    
    # Select implementations to run
    implementations = args.implementations if args.implementations else all_implementations
    datasets = ['tiny', 'small', 'medium', 'large','huge','gigantic']
    
    results = {}
    
    # Run tests for each implementation and dataset
    for impl in implementations:
        results[impl] = {}
        exec_path = f'./build/{impl}'
        
        for dataset in datasets:
            print(f"Running {impl} on {dataset} dataset...")
            execution_time = run_implementation(exec_path, dataset)
            
            if execution_time >= 0:
                results[impl][dataset] = execution_time
                print(f"{impl.capitalize()}, {dataset}: {execution_time:.6f} seconds")
    
    # Plot results
    plot_results(results)