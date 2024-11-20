import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Read output from C++ program
V = int(input())
graph = []
for _ in range(V):
    row = list(map(int, input().split()))
    graph.append(row)
cost = int(input())
path = list(map(int, input().split()))

# Create graph visualization
G = nx.Graph()
# Add edges with weights
edge_labels = {}
for i in range(V):
    for j in range(i+1, V):
        if graph[i][j] > 0:
            G.add_edge(i, j, weight=graph[i][j])
            edge_labels[(i,j)] = graph[i][j]

# Create layout
pos = nx.spring_layout(G)

# Draw the base graph
plt.figure(figsize=(10,10))
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=500, font_size=16, font_weight='bold')

# Draw edge labels (weights)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Draw path with arrows
path_edges = list(zip(path[:-1], path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                      edge_color='r', width=2,
                      arrows=True, arrowsize=20)

# Add sequence numbers along the path
for idx, node in enumerate(path[:-1]):
    x, y = pos[node]
    plt.annotate(f'#{idx+1}', 
                xy=(x, y+0.1),
                ha='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.title(f'TSP Path (Cost: {cost})')
plt.savefig('tsp_path.png', bbox_inches='tight')
plt.close()