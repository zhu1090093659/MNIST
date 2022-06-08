import torch

torch.cuda.empty_cache()
mem = torch.cuda.mem_get_info()
torch.cuda.max_memory_cached()
print(mem[1] / 1024 / 1024 - mem[0] / 1024 / 1024)
# print(str(mem[1] / (1024 * 1024 * 1024) - mem[0] / (1024 * 1024 * 1024) + 'GB'))
