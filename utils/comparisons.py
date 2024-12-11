#!/usr/bin/env python3
import subprocess
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.table import Table

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

def plot_table(results, datasets, implementations):
    """Generate a table of execution times with proper axis labels and save as an image"""
    data = []
    for dataset in datasets:
        row = []
        for impl in implementations:
            if dataset in results[impl]:
                row.append(f"{results[impl][dataset]:.2f}")
            else:
                row.append("")  # Leave blank if no data
        data.append(row)
    
    df = pd.DataFrame(data, columns=implementations, index=datasets)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add headers and cells with axis labels
    n_rows, n_cols = df.shape
    for i in range(n_rows + 1):
        for j in range(n_cols + 1):
            if i == 0 and j == 0:
                # Top-left corner: Leave blank
                text = ""
            elif i == 0:
                # Header row: Algorithms
                text = df.columns[j - 1]
            elif j == 0:
                # Header column: Datasets
                text = df.index[i - 1]
            else:
                # Data cells
                text = df.iloc[i - 1, j - 1]
            table.add_cell(
                i, j,
                width=1.0 / (n_cols + 1),
                height=0.2,
                text=text,
                loc='center',
                facecolor='white'
            )
    
    # Add X and Y axis labels
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.add_table(table)
    ax.text(0.5, 1.05, 'Algorithm', fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(-0.05, 0.5, 'Datasets', fontsize=12, va='center', rotation='vertical', transform=ax.transAxes)

    plt.savefig("execution_times_table.png", dpi=300, bbox_inches="tight")
    print("Execution times table saved as execution_times_table.png")

def plot_results(results):
    """Plot timing results with multiple views"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    colors = {
        'brute': 'b',         # Blue
        'greedy': 'g',        # Green
        'genetic': 'r',       # Red
        'dp': 'c',            # Cyan
        'greedy_cuda': 'y',   # Yellow
        'genetic_cuda': 'm',  # Magenta
        'dp_omp': 'orange',   # Orange
        'dp_cuda': 'purple'   # Purple
    }
    
    datasets = ['tiny', 'small', 'medium', 'large','huge','gigantic']
    dataset_sizes = [10, 27, 100, 1000, '10k', '100k']  # New tick labels
    x_pos = range(len(datasets))
    
    # Plot 1: Log scale (good for small differences)
    for impl, timings in results.items():
        if timings:
            filtered_datasets = [d for d in datasets if d in timings]  # Filter valid datasets for this implementation
            x_pos_filtered = [datasets.index(d) for d in filtered_datasets]  # Map filtered datasets to x_pos indices
            ax1.plot(x_pos_filtered, [timings[d] for d in filtered_datasets], 
                    f'{colors[impl]}-o', 
                    label=f'{impl.capitalize()}')
    ax1.set_yscale('log')
    ax1.set_title('Log Scale Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xlabel('Number of Cities')
    ax1.set_xticklabels(dataset_sizes) 
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Linear scale (good for large differences)
    for impl, timings in results.items():
        if timings:
            filtered_datasets = [d for d in datasets if d in timings]  # Filter valid datasets for this implementation
            x_pos_filtered = [datasets.index(d) for d in filtered_datasets]  # Map filtered datasets to x_pos indices
            ax2.plot(x_pos_filtered, [timings[d] for d in filtered_datasets], 
                    f'{colors[impl]}-o', 
                    label=f'{impl.capitalize()}')
    ax2.set_title('Linear Scale Comparison (Seconds)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dataset_sizes) 
    ax2.set_xlabel('Number of Cities')
    ax2.grid(True)
    
    # Plot 3: Speedup relative to baseline (greedy)
    if 'greedy' in results:
        baseline = results['greedy']
        for impl, timings in results.items():
            if impl != 'greedy' and timings:
                filtered_datasets = [d for d in datasets if d in timings and d in baseline]  # Filter valid datasets
                x_pos_filtered = [datasets.index(d) for d in filtered_datasets]  # Map filtered datasets to x_pos indices
                speedups = [baseline[d] / timings[d] for d in filtered_datasets]
                ax3.plot(x_pos_filtered, speedups, 
                        f'{colors[impl]}-o', 
                        label=f'{impl.capitalize()}')
        ax3.axhline(y=1.0, color='k', linestyle='--')
        ax3.set_title('Speedup vs. Greedy')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(dataset_sizes) 
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
    all_implementations = ['brute', 'greedy', 'genetic', 'dp', 'greedy_cuda']
    
    parser = argparse.ArgumentParser(description='Compare TSP implementations')
    parser.add_argument('implementations', 
                       nargs='*',
                       choices=all_implementations,
                       help='Implementations to compare. If none specified, runs all.')
    
    args = parser.parse_args()
    implementations = args.implementations if args.implementations else all_implementations
    datasets = ['tiny', 'small', 'medium', 'large','huge','gigantic']
    cutoff = {
        'brute': 'small',
        'dp': 'medium',
    }
    
    results = {}
    for impl in implementations:
        results[impl] = {}
        exec_path = f'./build/{impl}'
        
        for dataset in datasets:
            cutoff_point = cutoff.get(impl, None)
            if dataset == cutoff_point:
                break
            print(f"Running {impl} on {dataset} dataset...")
            execution_time = run_implementation(exec_path, dataset)
            if execution_time >= 0:
                results[impl][dataset] = execution_time
                print(f"{impl.capitalize()}, {dataset}: {execution_time:.6f} seconds")
    
    plot_results(results)
    plot_table(results, datasets, implementations)