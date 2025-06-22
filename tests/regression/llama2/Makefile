ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := llama2

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

TRACY_INC_DIR := /home/zhaodingyi/codes/tracy/public

CXXFLAGS += -I$(TRACY_INC_DIR) -DTRACY_ENABLE
SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/tracy/TracyClient.cpp

VX_SRCS := $(SRC_DIR)/matmul.cpp

OPTS ?= 

include ../common.mk
