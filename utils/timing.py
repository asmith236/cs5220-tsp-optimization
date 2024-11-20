import sys
import csv
import numpy as np
import subprocess
import time

sizes = [4, 5]
values = [
    [
        [0, 10, 15, 20],
        [10, 0, 25, 25],
        [15, 25, 0, 30],
        [20, 25, 30, 0]
    ],
    [
        [0, 10, 14, 7, 4],
        [10, 0, 15, 4, 9],
        [14, 15, 0, 13, 6],
        [7, 4, 13, 0, 30],
        [4, 9, 6, 30, 0]
    ]
]

def main(executable_name):
    times = np.zeros(len(sizes))

    for i in range(len(sizes)):
        dist = values[i]

        with open('dist_matrix.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(dist)

        # Record the start time for execution
        start_time = time.time()
        
        # Call the executable and capture the output
        result = subprocess.run([executable_name], capture_output=True, text=True, check=True)

        # Record the end time after execution
        end_time = time.time()        

        times[i] = end_time - start_time

    print(times)

if __name__ == "__main__":
    exec_name = sys.argv[1]

    main(exec_name)