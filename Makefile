CPP=g++
CFLAGS=-lm
OPTFLAGS=-O3
OMPFLAGS=-fopenmp 
NVCC= nvcc -arch=sm_70
NVCCFLAGS=-DCUDA

SRC_DIR=common
ALGO_DIR=algorithms
BUILD_DIR=build

all: brute 

brute: $(BUILD_DIR)/brute
dp: $(BUILD_DIR)/dp
greedy: $(BUILD_DIR)/greedy
genetic: $(BUILD_DIR)/genetic
greedy_cuda: $(BUILD_DIR)/greedy_cuda

$(BUILD_DIR)/brute: $(SRC_DIR)/main.cpp $(ALGO_DIR)/brute.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/dp: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/greedy: $(SRC_DIR)/main.cpp $(ALGO_DIR)/greedy.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/genetic: $(SRC_DIR)/main.cpp $(ALGO_DIR)/genetic.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/greedy_cuda: $(SRC_DIR)/main.cpp $(ALGO_DIR)/greedy_cuda.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

.PHONY: clean

clean:
	rm -f $(BUILD_DIR)/*
