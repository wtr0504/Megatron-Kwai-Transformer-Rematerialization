#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-7b

export SEQ_LENGTH=16384
export GLOBAL_BATCH_SIZE=16

export HOSTFILE=
export MASTER_ADDR=127.0.0.1
export NUM_GPUS=1

export TP=1
export CP=1
export PP=1
export PP_l=1
export CKPT=full
export OFFLOAD_ALPHA=0.0
export CUDA_VISIBLE_DEVICES=7
./pretrain_llama.sh
 