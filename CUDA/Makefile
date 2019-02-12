################################################################################
# Makefile for general code snippets
#
# by André Pereira (LIP-Minho)
################################################################################
SHELL = /bin/sh

BIN_NAME = simple

#CXX = g++
#LD  = g++

detected_OS := $(shell sh -c 'uname -s 2>/dev/null || echo not')
CXX = nvcc
LD  = nvcc

#CXXFLAGS += -O3 -DD_GPU
CXXFLAGS += -g -G -DD_GPU
CXXFLAGS += -I/share/apps/papi/5.4.1/include -L/share/apps/papi/5.4.1/lib  -lpapi
################################################################################
# Control awesome stuff
################################################################################

SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build
SRC = $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(patsubst src/%.cu,build/%.o,$(SRC))
DEPS = $(patsubst build/%.o,build/%.d,$(OBJ))
BIN = $(BIN_NAME)

vpath %.cu $(SRC_DIR)

################################################################################
# Rules
################################################################################

.DEFAULT_GOAL = all

$(BUILD_DIR)/%.d: %.cu
	$(CXX) -M $(CXXFLAGS) $(INCLUDES) $< -o $@ 

$(BUILD_DIR)/%.o: %.cu
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@ 

$(BIN_DIR)/$(BIN_NAME): $(DEPS) $(OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJ) 

checkdirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

all: checkdirs $(BIN_DIR)/$(BIN_NAME)

clean:
	rm -f $(BUILD_DIR)/* $(BIN_DIR)/*
