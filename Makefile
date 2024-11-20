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

all:	$(BUILD_DIR) $(addprefix $(BUILD_DIR)/, $(TARGETS))

$(BUILD_DIR)/basic: basic.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/serial: serial.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean