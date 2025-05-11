import subprocess

def get_free_gpu_memory():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        free_memory_list = [int(x) for x in result.strip().split('\n')]
        return free_memory_list
    except Exception as e:
        print("获取 GPU 显存失败:", e)
        return []

def get_max_free_memory_gpu():
    free_memory_list = get_free_gpu_memory()
    if not free_memory_list:
        return None
    max_mem = max(free_memory_list)
    gpu_index = free_memory_list.index(max_mem)
    return gpu_index, max_mem

if __name__ == "__main__":
    gpu_index, free_mem = get_max_free_memory_gpu()
    if gpu_index is not None:
        print(f"剩余显存最多的是 GPU {gpu_index}，剩余 {free_mem} MB")
    else:
        print("未检测到可用 GPU")
