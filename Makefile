# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall -std=c99

# Target executable
TARGET = tsp

# Source files
SRCS = tsp.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default target
all: karp_basic

# Linking the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Compiling source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
