#!/usr/bin/env python3
import subprocess
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

def run_implementation(exec_path: str, dataset: str) -> float:
    try:
        start = time.time()
        result = subprocess.run([exec_path, '--csv', f'data/{dataset}.csv'], 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        return time.time() - start
    except Exception as e:
        print(f"Error running {exec_path} with {dataset}: {e}")
        return -1

def plot_results(results):
    """Plot timing results with multiple views"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    colors = {
        'brute': 'b',
        'greedy': 'g',
        'genetic': 'r',
        'dp': 'g',
        'dp_omp': 'y'
    }
    
    datasets = ['5', '10', '15', '20','25']
    x_pos = range(len(datasets))
    
    # Plot 1: Log scale (good for small differences)
    for impl, timings in results.items():
        if timings:
            ax1.plot(x_pos, [timings[d] for d in datasets], 
                    f'{colors[impl]}-o', 
                    label=f'{impl.capitalize()}')
    ax1.set_yscale('log')
    ax1.set_title('Log Scale Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets)
    ax1.set_xlabel('Number of Cities')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Linear scale (good for large differences)
    for impl, timings in results.items():
        if timings:
            ax2.plot(x_pos, [timings[d] for d in datasets], 
                    f'{colors[impl]}-o', 
                    label=f'{impl.capitalize()}')
    ax2.set_title('Linear Scale Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xlabel('Number of Cities')
    ax2.set_xticklabels(datasets)
    ax2.grid(True)
    
    # Plot 3: Speedup relative to baseline (greedy)
    if 'dp' in results:
        baseline = results['dp']
        for impl, timings in results.items():
            if impl != 'dp' and timings:
                speedups = [baseline[d]/timings[d] for d in datasets]
                ax3.plot(x_pos, speedups, 
                        f'{colors[impl]}-o', 
                        label=f'{impl.capitalize()}')
        ax3.axhline(y=1.0, color='k', linestyle='--')
        ax3.set_title('Speedup vs. Greedy')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(datasets)
        ax3.set_xlabel('Number of Cities')
        ax3.grid(True)
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as comparison_results.png")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")

if __name__ == "__main__":
    all_implementations = ['brute', 'greedy', 'genetic', 'dp', 'greedy_cuda', 'dp_omp']
    
    parser = argparse.ArgumentParser(description='Compare TSP implementations')
    parser.add_argument('implementations', 
                       nargs='*',
                       choices=all_implementations,
                       help='Implementations to compare. If none specified, runs all.')
    
    args = parser.parse_args()
    implementations = args.implementations if args.implementations else all_implementations
    datasets = ['5', '10', '15', '20','25']
    
    results = {}
    for impl in implementations:
        results[impl] = {}
        exec_path = f'./build/{impl}'
        
        for dataset in datasets:
            print(f"Running {impl} on {dataset} dataset...")
            execution_time = run_implementation(exec_path, dataset)
            if execution_time >= 0:
                results[impl][dataset] = execution_time
                print(f"{impl.capitalize()}, {dataset}: {execution_time:.6f} seconds")
    
    plot_results(results)