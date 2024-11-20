# Compiler and flags
CPP = g++
CFLAGS = -lm
COPTFLAGS = -O3 -Wall

# Directories
BUILD_DIR = build

# Source file
SRC = karp_basic.cpp

# Target executable
TARGET = $(BUILD_DIR)/karp_basic

# Default target
all: $(TARGET)

# Building the target executable
$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(CPP) $(SRC) -o $(TARGET) $(CFLAGS) $(COPTFLAGS)

# Clean up
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
