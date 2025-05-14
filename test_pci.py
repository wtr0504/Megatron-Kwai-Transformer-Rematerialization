import torch
import time
def measure_copy_bandwidth(size_mb=512):
    tensor = torch.empty(size_mb * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    t_cuda = torch.empty_like(tensor, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    t_cuda.copy_(tensor, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{size_mb} MB copy took {elapsed * 1000:.2f} ms → Bandwidth: {size_mb / elapsed:.2f} MB/s")

measure_copy_bandwidth(512)


size = 100 * 1024 * 1024 * 100  # 100MB float32
cpu_tensor = torch.empty(size // 4, dtype=torch.float32, pin_memory=True)
gpu_tensor = torch.empty_like(cpu_tensor, device='cuda')

stream = torch.cuda.Stream()

start = time.time()
with torch.cuda.stream(stream):
    gpu_tensor.copy_(cpu_tensor, non_blocking=True)

elapsed = time.time() - start
print(f"CPU time after copy call: {elapsed * 1000:.3f} ms")  # 应该很小，几百微秒以内

torch.cuda.synchronize(stream)  # 再等实际拷贝完成
print(f"Total copy time: {(time.time() - start) * 1000:.3f} ms")
