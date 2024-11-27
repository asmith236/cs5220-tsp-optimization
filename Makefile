CPP=g++
CFLAGS=-lm -lnuma
OPTFLAGS=-O3 -march=native -mtune=native -fopenmp -ffast-math -funroll-loops -floop-parallelize-all -lm

SRC_DIR=common
ALGO_DIR=algorithms
BUILD_DIR=build

all: brute dp genetic dp_omp

brute: $(BUILD_DIR)/brute
dp: $(BUILD_DIR)/dp
greedy: $(BUILD_DIR)/greedy
genetic: $(BUILD_DIR)/genetic
dp_omp: $(BUILD_DIR)/dp_omp

$(BUILD_DIR)/brute: $(SRC_DIR)/main.cpp $(ALGO_DIR)/brute.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/dp: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/greedy: $(SRC_DIR)/main.cpp $(ALGO_DIR)/greedy.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/genetic: $(SRC_DIR)/main.cpp $(ALGO_DIR)/genetic.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

$(BUILD_DIR)/dp_omp: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp_omp.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

.PHONY: clean

clean:
	rm -f $(BUILD_DIR)/*
