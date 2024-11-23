# cs5220-tsp-optimization

This CS 5220 final project aims to optimize the traveling salesman problem using dynamic programming and other high-performance computing techniques.

## Methods to Run

make all
./build/brute --csv tiny.csv
make clean

# For visualizer

module load python
python utils/visualizer.py tiny build/brute.out

# References

https://www.kaggle.com/datasets/mexwell/traveling-salesman-problem/data