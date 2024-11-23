#!/usr/bin/env python3
import subprocess
import time
import argparse
import glob
import matplotlib.pyplot as plt
import re

def get_num_cities(filename):
    """Extract number of cities from test case filename"""
    return int(re.search(r'test_case_(\d+).txt', filename).group(1))

def run_implementation(exec_path: str, input_file: str) -> float:
    """Run implementation with input file and return execution time"""
    try:
        start = time.time()
        result = subprocess.run([exec_path, input_file], 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        end = time.time()
        return end - start
    except Exception as e:
        print(f"Error running {exec_path} with {input_file}: {e}")
        return -1

def plot_results(results, show_basic=True, show_serial=True):
    """Plot timing results and save to file"""
    plt.figure(figsize=(10, 6))
    
    if show_basic and 'basic' in results:
        cities, times = zip(*sorted(results['basic'].items()))
        plt.plot(cities, times, 'b-o', label='Basic Implementation')
    
    if show_serial and 'serial' in results:
        cities, times = zip(*sorted(results['serial'].items()))
        plt.plot(cities, times, 'r-o', label='Serial Implementation')
    
    plt.xlabel('Number of Cities')
    plt.ylabel('Execution Time (seconds)')
    plt.title('TSP Implementation Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    # Save plot to file
    plt.savefig('timing_results.png')
    print("Plot saved as timing_results.png")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time TSP implementations')
    parser.add_argument('implementation', 
                       nargs='?',
                       choices=['basic', 'serial'],
                       help='Implementation to time (basic or serial). If omitted, runs both.')
    
    args = parser.parse_args()
    
    # Get all test case files
    test_files = sorted(glob.glob('test_case_*.txt'), 
                       key=get_num_cities)
    
    results = {}
    implementations = ['basic', 'serial'] if not args.implementation else [args.implementation]
    
    # Run tests for each implementation
    for impl in implementations:
        results[impl] = {}
        exec_path = f'./build/{impl}'
        
        for test_file in test_files:
            num_cities = get_num_cities(test_file)
            execution_time = run_implementation(exec_path, test_file)
            
            if execution_time >= 0:
                results[impl][num_cities] = execution_time
                print(f"{impl.capitalize()} implementation, {num_cities} cities: {execution_time:.6f} seconds")
    
    # Plot results
    plot_results(results, 
                show_basic=('basic' in implementations),
                show_serial=('serial' in implementations))