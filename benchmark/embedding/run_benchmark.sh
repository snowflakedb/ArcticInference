#!/bin/bash

CURR_DIR=$(pwd)
FILE_DIR=$(dirname $0)
MODEL=${1:-"Snowflake/snowflake-arctic-embed-m-v1.5"}
N_REQUESTS=${2:-1280}
PROMPT_LENGTH=${3:-50}
CONCURRENCY=${4:-64}
DISTRIBUTION=${5:-"fixed"}
BATCH_SIZE=${6:-1,4,16,64}
NUM_REPLICAS=${7:-4}

function run_benchmark_vllm() {
    echo "Running vllm benchmark for $MODEL"
    pkill -f vllm
    vllm serve $MODEL --port 8000 > vllm.log &
    pid=$!
    sleep 60
    python ${FILE_DIR}/benchmark_http.py --model $MODEL \
    --server http://localhost:8000 \
    --batch-sizes $BATCH_SIZE \
    --requests $N_REQUESTS \
    --concurrency $CONCURRENCY \
    --prompt-length $PROMPT_LENGTH \
    --distribution $DISTRIBUTION
    kill $pid
    pkill -f vllm
}

function run_benchmark_arctic() {
    echo "Running arctic_inference benchmark for $MODEL"
    pkill -f replica.py
    python -m arctic_inference.embedding.replica_manager --model $MODEL --num-replicas $NUM_REPLICAS --port 50050 > arctic.log &
    pid=$!
    sleep 20
    python ${FILE_DIR}/benchmark.py --model $MODEL \
    --server localhost:50050 \
    --batch-sizes $BATCH_SIZE \
    --requests $N_REQUESTS \
    --concurrency $CONCURRENCY \
    --prompt-length $PROMPT_LENGTH \
    --distribution $DISTRIBUTION
    kill $pid
    pkill -f python
}

function setup() {
    echo "Install packages and generate gRPC code"
    pip install -U grpcio grpcio-tools protobuf > benchmark.log 2>&1;
    cd ${FILE_DIR}/../;
    python arctic_inference/embedding/generate_proto.py >> benchmark.log 2>&1;
    cd ${CURR_DIR};
}


run_benchmark_vllm
run_benchmark_arctic
