# sudo docker run --gpus all -it --rm \
#   --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
#   -e NCCL_DEBUG=INFO \
#   -e NCCL_IB_DISABLE=1 \
#   -e NCCL_P2P_DISABLE=1 \
#   -e NCCL_SOCKET_IFNAME=eth0 \ # 或者你容器实际用的网卡名
#   -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
#   -v /vepfs/home/wangtaoran/Megatron-Kwai:/root/Megatron-Kwai \
#   yuantailing/megatron-kwai:atc24ae-1.0.0 /bin/bash

sudo docker run -it --name ppoffload --gpus all \
    --shm-size 32g \
   --rm -v  /vepfs/home/wangtaoran/Megatron-Kwai:/root/Megatron-Kwai \
    yuantailing/megatron-kwai:atc24ae-1.0.0  \
    /bin/bash
# 运行容器后，执行以下命令
# export NCCL_DEBUG=INFO 
# export NCCL_IB_DISABLE=1 
# export NCCL_P2P_DISABLE=1 
# export NCCL_SOCKET_IFNAME=$(ip route get 8.8.8.8 | awk '{print $5; exit}') 
# export TORCH_DISTRIBUTED_DEBUG=DETAIL 
# export CUDA_VISIBLE_DEVICES=0,1,2,3

sudo docker run -it --name ppoffload --gpus all \
    --shm-size 32g \
   --rm -v  /vepfs/home/wangtaoran/Megatron-Kwai:/root/Megatron-Kwai \
    yuantailing/megatron-kwai:atc24ae-1.0.0  \
    /bin/bash