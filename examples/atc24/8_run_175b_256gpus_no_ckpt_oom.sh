#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-175b

export SEQ_LENGTH=8192
export GLOBAL_BATCH_SIZE=256

export HOSTFILE=/etc/mpi/hostfile
export MASTER_ADDR=$MY_NODE_IP
export NUM_GPUS=256

export TP=2
export CP=2
export PP=16
export PP_l=1
export CKPT=no
export OFFLOAD_ALPHA=0

./pretrain_llama.sh
