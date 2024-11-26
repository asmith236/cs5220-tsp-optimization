CPP=g++
NVCC=nvcc
CFLAGS=-lm
OPTFLAGS=-O3 

SRC_DIR=common
ALGO_DIR=algorithms
BUILD_DIR=build

all: brute dp genetic cu_genetic

brute: $(BUILD_DIR)/brute
dp: $(BUILD_DIR)/dp
greedy: $(BUILD_DIR)/greedy
genetic: $(BUILD_DIR)/genetic
cu_genetic: $(BUILD_DIR)/cu_genetic

$(BUILD_DIR)/brute: $(SRC_DIR)/main.cpp $(ALGO_DIR)/brute.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/dp: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/greedy: $(SRC_DIR)/main.cpp $(ALGO_DIR)/greedy.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/genetic: $(SRC_DIR)/main.cpp $(ALGO_DIR)/genetic.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/cu_genetic: $(SRC_DIR)/main.cpp $(ALGO_DIR)/genetic.cu
	$(NVCC) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

.PHONY: clean

clean:
	rm -f $(BUILD_DIR)/*
