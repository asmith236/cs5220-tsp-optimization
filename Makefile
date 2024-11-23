CXX = g++
CXXFLAGS = -Wall -O3 -std=c++11

# Directory structure
BUILD_DIR = build
SRC_DIR = .

# Targets
TARGETS = basic serial

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: $(BUILD_DIR) $(addprefix $(BUILD_DIR)/, $(TARGETS))

# Build the basic target
$(BUILD_DIR)/basic: basic.cpp $(SRC_DIR)/common/constants.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ basic.cpp $(SRC_DIR)/common/constants.cpp

# Build the serial target
$(BUILD_DIR)/serial: serial.cpp $(SRC_DIR)/common/constants.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ serial.cpp $(SRC_DIR)/common/constants.cpp

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
