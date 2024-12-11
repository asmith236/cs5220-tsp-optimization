CPP=g++
NVCC= nvcc -arch=sm_70
NVCCFLAGS=-DCUDA
CFLAGS=-lm
OPTFLAGS=-O3 

CFLAGS_DP=-lm
OPTFLAGS_DP=-O3 -march=native -mtune=native -fopenmp -ffast-math -funroll-loops -floop-parallelize-all 

CFLAGS_NUMA=-lm -lnuma

SRC_DIR=common
ALGO_DIR=algorithms
BUILD_DIR=build

all: brute dp genetic greedy genetic_cuda dp_omp dp_cuda greedy_cuda dp_numa

brute: $(BUILD_DIR)/brute
dp: $(BUILD_DIR)/dp
greedy: $(BUILD_DIR)/greedy
genetic: $(BUILD_DIR)/genetic
genetic_cuda: $(BUILD_DIR)/genetic_cuda
dp_omp: $(BUILD_DIR)/dp_omp
greedy_cuda: $(BUILD_DIR)/greedy_cuda
dp_cuda: $(BUILD_DIR)/dp_cuda
dp_numa: $(BUILD_DIR)/dp_numa

$(BUILD_DIR)/brute: $(SRC_DIR)/main.cpp $(ALGO_DIR)/brute.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/dp: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/greedy: $(SRC_DIR)/main.cpp $(ALGO_DIR)/greedy.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/genetic: $(SRC_DIR)/main.cpp $(ALGO_DIR)/genetic.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/genetic_cuda: $(SRC_DIR)/main.cpp $(ALGO_DIR)/genetic_cuda.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

$(BUILD_DIR)/dp_omp: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp_omp.cpp
	$(CPP) $^ -o $@ $(CFLAGS_DP) $(OPTFLAGS_DP)

$(BUILD_DIR)/greedy_cuda: $(SRC_DIR)/main.cpp $(ALGO_DIR)/greedy_cuda.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

$(BUILD_DIR)/dp_cuda: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp_cuda.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

$(BUILD_DIR)/dp_numa: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp_numa.cpp
	$(CPP) $^ -o $@ $(CFLAGS_NUMA) $(OPTFLAGS_DP)

.PHONY: clean

clean:
	rm -f $(BUILD_DIR)/*
