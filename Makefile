#!/bin/bash
ITER?=64

SRC = ./src/main.cpp
OUT = ./flops_test

ROCM_PATH?="/opt/rocm"
CXX=${ROCM_PATH}/bin/hipcc

$(OUT): $(SRC)
	${CXX} -std=c++17 -D ITER=$(ITER) -w -o $(OUT) $(SRC)

.PHONY: all
all: $(OUT)

clean:
	rm ${OUT}