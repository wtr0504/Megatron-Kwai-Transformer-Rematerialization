export NCCL_DEBUG=INFO \
export NCCL_IB_DISABLE=1 \
export NCCL_P2P_DISABLE=1 \
export NCCL_SOCKET_IFNAME=$(ip route get 8.8.8.8 | awk '{print $5; exit}') \
export TORCH_DISTRIBUTED_DEBUG=DETAIL \
export CUDA_VISIBLE_DEVICES=0,1,6,7