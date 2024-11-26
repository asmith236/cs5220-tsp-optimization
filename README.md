# cs5220-tsp-optimization

This CS 5220 final project aims to optimize the traveling salesman problem using dynamic programming and other high-performance computing techniques.

# Methods to Run

make all  
./build/brute --csv data/tiny.csv  
make clean  

# For Correctness

python utils/correctness.py --alg brute --csv tiny

# For visualizer

module load python
python utils/visualizer.py tiny build/brute.out

# For Timing Comparison Graph

module load python
python utils/comparisons.py (tries all implementations)
python utils/comparisons.py greedy dp (only compares greedy and dp)

# References

https://www.kaggle.com/datasets/mexwell/traveling-salesman-problem/data