# Directories
SRC_DIR = src
TEST_DIR = tests
OBJ_DIR = obj
BIN_DIR = bin

# Shared sources
COMMON_SRC = \
	$(SRC_DIR)/mlpnet.c \
	$(SRC_DIR)/helpers.c

# Compiler and flags
CC = gcc
CFLAGS = -Wall -g -I$(SRC_DIR)

# Binaries
BINARIES = xor lotus

# Targets for building both
all: $(BINARIES)

# Create required directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/$(SRC_DIR) $(OBJ_DIR)/$(TEST_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# ----- XOR TARGET -----

XOR_SRC = $(COMMON_SRC) $(TEST_DIR)/xor_mlp_test.c
XOR_OBJ = $(patsubst %.c,$(OBJ_DIR)/%.o,$(XOR_SRC))

xor: $(OBJ_DIR) $(BIN_DIR) $(BIN_DIR)/xor

$(BIN_DIR)/xor: $(XOR_OBJ)
	$(CC) $(CFLAGS) -o $@ $(XOR_OBJ)

# ----- LOTUS TARGET -----

LOTUS_SRC = $(COMMON_SRC) $(TEST_DIR)/lotus_mlp_test.c
LOTUS_OBJ = $(patsubst %.c,$(OBJ_DIR)/%.o,$(LOTUS_SRC))

lotus: $(OBJ_DIR) $(BIN_DIR) $(BIN_DIR)/lotus

$(BIN_DIR)/lotus: $(LOTUS_OBJ)
	$(CC) $(CFLAGS) -o $@ $(LOTUS_OBJ)

# ----- Compilation Rule for All .c Files -----

$(OBJ_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ----- Cleanup -----

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all xor lotus clean