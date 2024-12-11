#!/usr/bin/env python3
import subprocess
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.table import Table
import os

def create_subset_csv(input_csv: str, output_csv: str, num_rows: int):
    """Create a new CSV with the first `num_rows` rows of the input CSV."""
    try:
        df = pd.read_csv(input_csv)
        df_subset = df.head(num_rows)
        df_subset.to_csv(output_csv, index=False)
        # print(f"Created subset CSV with the first {num_rows} rows: {output_csv}")
    except Exception as e:
        print(f"Error creating subset CSV: {e}")

def run_implementation(exec_path: str, dataset: str) -> float:
    """Run the implementation on the given dataset."""
    try:
        start = time.time()
        result = subprocess.run([exec_path, '--csv', dataset], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        return time.time() - start
    except Exception as e:
        print(f"Error running {exec_path} on {dataset}: {e}")
        return -1

def plot_table(results, dataset_sizes, implementations):
    """Generate a table of execution times with proper axis labels and save as an image"""
    data = []
    for dataset in dataset_sizes:
        row = []
        for impl in implementations:
            if dataset in results[impl]:
                row.append(f"{results[impl][dataset]:.2f}")
            else:
                row.append("")  # Leave blank if no data
        data.append(row)
    
    df = pd.DataFrame(data, columns=implementations, index=dataset_sizes)
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
    ax.text(-0.05, 0.5, 'Number of Cities', fontsize=12, va='center', rotation='vertical', transform=ax.transAxes)

    plt.savefig("execution_times_table.png", dpi=300, bbox_inches="tight")
    print("Execution times table saved as execution_times_table.png")

def plot_results(results):
    """Plot timing results with multiple views"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    colors = {
        'brute': 'b',          # Blue
        'greedy': 'g',         # Green
        'genetic': 'r',        # Red
        'dp': 'c',             # Cyan
        'greedy_cuda': 'y',    # Yellow
        'genetic_cuda': 'm',   # Magenta
        'dp_omp': '#FFA500',   # Orange (RGB Hex Code)
        'dp_cuda': '#800080',  # Purple (RGB Hex Code)
        'dp_numa': '#008080'   # Teal
    }

    dataset_sizes = [11, 13, 15, 17, 19, 21, 23, 25, 27]  # Number of cities
    x_pos = range(len(dataset_sizes))  # Positions for X-axis

    # Plot 1: Log scale (good for small differences)
    for impl, timings in results.items():
        if timings:
            filtered_sizes = [size for size in dataset_sizes if size in timings]  # Filter valid dataset sizes
            x_pos_filtered = [dataset_sizes.index(size) for size in filtered_sizes]  # Map sizes to X positions
            ax1.plot(x_pos_filtered, [timings[size] for size in filtered_sizes], 
                     '-o', color=colors[impl], label=f'{impl.capitalize()}')  # Use explicit color
    ax1.set_yscale('log')
    ax1.set_title('Log Scale Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dataset_sizes)
    ax1.set_xlabel('Number of Cities')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Linear scale (good for large differences)
    for impl, timings in results.items():
        if timings:
            filtered_sizes = [size for size in dataset_sizes if size in timings]  # Filter valid dataset sizes
            x_pos_filtered = [dataset_sizes.index(size) for size in filtered_sizes]  # Map sizes to X positions
            ax2.plot(x_pos_filtered, [timings[size] for size in filtered_sizes], 
                     '-o', color=colors[impl], label=f'{impl.capitalize()}')  # Use explicit color
    ax2.set_title('Linear Scale Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dataset_sizes)
    ax2.set_xlabel('Number of Cities')
    ax2.grid(True)

    # Plot 3: Speedup relative to baseline (greedy)
    if 'dp' in results:
        baseline = results['dp']
        for impl, timings in results.items():
            if impl != 'dp' and timings:
                filtered_sizes = [size for size in dataset_sizes if size in timings and size in baseline]
                x_pos_filtered = [dataset_sizes.index(size) for size in filtered_sizes]
                speedups = [baseline[size] / timings[size] for size in filtered_sizes]
                ax3.plot(x_pos_filtered, speedups, 
                         '-o', color=colors[impl], label=f'{impl.capitalize()}')  # Use explicit color
        ax3.axhline(y=1.0, color='k', linestyle='--')
        ax3.set_title('Speedup vs. DP')
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
    all_implementations = ['brute', 'greedy', 'genetic', 'dp', 'greedy_cuda', 'genetic_cuda', 'dp_omp', 'dp_cuda', 'dp_numa']
    
    parser = argparse.ArgumentParser(description='Compare TSP implementations')
    parser.add_argument('implementations', 
                       nargs='*',
                       choices=all_implementations,
                       help='Implementations to compare. If none specified, runs all.')
    
    args = parser.parse_args()
    implementations = args.implementations if args.implementations else all_implementations

    input_csv = 'data/small.csv'  # Source CSV file
    temp_csv = 'data/temp_subset.csv'  # Temporary subset CSV file
    dataset_sizes = [11, 13, 15, 17, 19, 21, 23, 25, 27]  # Number of rows to use

    results = {}
    for impl in implementations:
        results[impl] = {}
        exec_path = f'./build/{impl}'
        
        for size in dataset_sizes:
            # Create a subset CSV with the first `size` rows
            create_subset_csv(input_csv, temp_csv, size-1)

            # Run the implementation on the subset CSV
            print(f"Running {impl} on first {size} rows of the small dataset...")
            execution_time = run_implementation(exec_path, temp_csv)
            if execution_time >= 0:
                results[impl][size] = execution_time
                print(f"{impl.capitalize()}, {size} cities: {execution_time:.6f} seconds")
    
    # Clean up the temporary CSV file
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
        print(f"Temporary file {temp_csv} removed.")
    
    # Generate the results table
    plot_results(results)
    plot_table(results, dataset_sizes, implementations)