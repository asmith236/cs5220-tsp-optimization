import pandas as pd
import numpy as np
import random
from sys import maxsize 
import matplotlib.cm as cm
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance_matrix
import plotly.graph_objects as go

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def print_best_route(best_path):
    plt.figure(figsize=(8,8))
    for i, index in enumerate(best_path):
        x, y = coordinates[index]
        plt.text(x, y, str(i+1), fontsize=12, color='black', ha='right', va='top')
    plt.plot([coordinates[i][0] for i in best_path], [coordinates[i][1] for i in best_path], 'ro-')
    plt.plot([coordinates[best_path[-1]][0], coordinates[best_path[0]][0]], [coordinates[best_path[-1]][1], coordinates[best_path[0]][1]], 'ro-')
    plt.title('Best Path for the Salesman')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def tsp_dp(coordinates):
    n = len(coordinates)
    distances = [[distance(p1, p2) for p2 in coordinates] for p1 in coordinates]
    
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0 
    
    for mask in range(1 << n):
        for u in range(n):
            if (mask & (1 << u)) == 0: 
                continue
            for v in range(n):
                if (mask & (1 << v)) == 0:
                    continue
                dp[mask][u] = min(dp[mask][u], dp[mask ^ (1 << u)][v] + distances[v][u])

    min_distance = float('inf')
    for u in range(1, n):
        min_distance = min(min_distance, dp[(1 << n) - 1][u] + distances[u][0])
    
    mask = (1 << n) - 1
    last = 0
    best_path = [0]
    for _ in range(n - 1):
        next_city = min(range(n), key=lambda v: dp[mask][v] + distances[v][last] if mask & (1 << v) else float('inf'))
        best_path.append(next_city)
        mask ^= (1 << next_city)
        last = next_city
    best_path.append(0)
    return best_path, min_distance

small = pd.read_csv('data/small_20.csv', names=['A', 'B'], header=None)
coordinates = small[['A', 'B']].values.tolist()

best_path, min_distance = tsp_dp(coordinates)

print("Best path:", best_path)
print("Minimum distance:", min_distance)

# print_best_route(best_path)
