#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-65b

export SEQ_LENGTH=8192
export GLOBAL_BATCH_SIZE=256

export HOSTFILE=/etc/mpi/hostfile
export MASTER_ADDR=$MY_NODE_IP
export NUM_GPUS=256

export TP=2
export CP=2
export PP=8
export PP_l=2
export CKPT=ours
export OFFLOAD_ALPHA=0.20

./pretrain_llama.sh
