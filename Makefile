CXX = g++
CXXFLAGS = -Wall -O3 -std=c++11

# Directory structure
BUILD_DIR = build
SRC_DIR = .

# Targets
TARGETS = brute dp

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: $(BUILD_DIR) $(addprefix $(BUILD_DIR)/, $(TARGETS))

# Build the brute target
$(BUILD_DIR)/brute: brute.cpp $(SRC_DIR)/common/constants.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ brute.cpp $(SRC_DIR)/common/constants.cpp

# Build the dp target
$(BUILD_DIR)/dp: dp.cpp $(SRC_DIR)/common/constants.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ dp.cpp $(SRC_DIR)/common/constants.cpp

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
