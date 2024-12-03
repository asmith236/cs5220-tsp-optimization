CPP=g++
CFLAGS=-lm
OPTFLAGS=-O3 

NVCC=nvcc
NVCCFLAGS=-DCUDA
# NVCCFLAGS=-DCUDA -O3 -dlto -use_fast_math -Xptxas -dlcm=ca -Xcompiler "-funroll-loops"

SRC_DIR=common
ALGO_DIR=algorithms
BUILD_DIR=build

all: brute dp dp-cuda

brute: $(BUILD_DIR)/brute
dp: $(BUILD_DIR)/dp
greedy: $(BUILD_DIR)/greedy
genetic: $(BUILD_DIR)/genetic
dupa-cuda: $(BUILD_DIR)/dp-cuda

$(BUILD_DIR)/brute: $(SRC_DIR)/main.cpp $(ALGO_DIR)/brute.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/dp: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/greedy: $(SRC_DIR)/main.cpp $(ALGO_DIR)/greedy.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/genetic: $(SRC_DIR)/main.cpp $(ALGO_DIR)/genetic.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/dp-cuda: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp-cuda.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

.PHONY: clean

clean:
	rm -f $(BUILD_DIR)/*
