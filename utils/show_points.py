import matplotlib.pyplot as plt
import pandas as pd
import os

# File path for the small30.csv relative to the project structure
file_path = os.path.join(os.path.dirname(__file__), '../data/small30.csv')

# Load the coordinates from the CSV file
data = pd.read_csv(file_path, header=None)
data.columns = ['x', 'y']  # Assign meaningful column names

# Extract x and y coordinates
x_coords = data['x']
y_coords = data['y']

# Create a scatter plot of the nodes
plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, color='blue', s=50, label='Nodes')

# Add labels to each node
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    plt.text(x, y, str(i), fontsize=9, ha='right', va='bottom')

# Add axis labels and a title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Visualization of Nodes with Labels')
plt.grid(True)
plt.legend()

# Save the plot to a file
output_path = os.path.join(os.path.dirname(__file__), '../data/small30_visualization.png')
plt.savefig(output_path, format='png')
print(f"Visualization saved as {output_path}")
