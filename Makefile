CPP=g++
CFLAGS=-lm
OPTFLAGS=-O3 

NVCC= nvcc -arch=sm_70
NVCCFLAGS=-DCUDA

CFLAGS_DP=-lm -lnuma
OPTFLAGS_DP=-O3 -march=native -mtune=native -fopenmp -ffast-math -funroll-loops -floop-parallelize-all -lm

MPI_COMPILER=mpicxx
MPI_FLAGS=$(CFLAGS) -lmpi -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include

GTL_LIB_PATH=/opt/cray/pe/mpich/8.1.28/gtl/lib  # Replace <version> with your specific version
GTL_FLAGS=-L$(GTL_LIB_PATH) -lmpi_gtl_cuda

SRC_DIR=common
ALGO_DIR=algorithms
BUILD_DIR=build

all: brute dp genetic greedy genetic_cuda dp_omp dp_cuda greedy_cuda dp_cuda_mpi

brute: $(BUILD_DIR)/brute
dp: $(BUILD_DIR)/dp
greedy: $(BUILD_DIR)/greedy
genetic: $(BUILD_DIR)/genetic
genetic_cuda: $(BUILD_DIR)/genetic_cuda
dp_omp: $(BUILD_DIR)/dp_omp
greedy_cuda: $(BUILD_DIR)/greedy_cuda
dp_cuda: $(BUILD_DIR)/dp_cuda
dp_cuda_mpi: $(BUILD_DIR)/dp_cuda_mpi

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

$(BUILD_DIR)/dp_cuda_mpi: $(SRC_DIR)/main.cpp $(ALGO_DIR)/dp_cuda_mpi.cu
	nvcc -ccbin $(MPI_COMPILER) $^ -o $@ $(MPI_FLAGS) $(GTL_FLAGS)

.PHONY: clean

clean:
	rm -f $(BUILD_DIR)/*
