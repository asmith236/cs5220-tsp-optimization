# cs5220-tsp-optimization

This CS 5220 final project aims to optimize the traveling salesman problem using dynamic programming and other high-performance computing techniques.

## Methods to Run

make all  
./build/brute --csv data/tiny.csv  
make clean  

## CUDA Configuration

salloc --nodes 4 --ntasks-per-node=128 --qos interactive --time 01:00:00 --constraint gpu --account m4776

## For Correctness

python utils/correctness.py --alg brute --csv tiny

## For visualizer

module load python  
python utils/visualizer.py tiny build/brute.out  

## For Timing Comparison Graph

module load python  
python utils/comparisons.py (tries all implementations)  
python utils/comparisons.py greedy dp (only compares greedy and dp)  

## References

Kaggle Dataset: https://www.kaggle.com/datasets/mexwell/traveling-salesman-problem/data  
Mexwell Jupyter Notebook: https://www.kaggle.com/code/mexwell/solving-traveling-salesman-problem