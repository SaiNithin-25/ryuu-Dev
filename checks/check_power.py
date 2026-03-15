import torch
import psutil
import platform

print("=== System Info ===")
print(f"OS: {platform.system()} {platform.release()}")
print(f"CPU Cores: {psutil.cpu_count(logical=False)} | Threads: {psutil.cpu_count(logical=True)}")
print(f"RAM: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")

print("=== PyTorch Info ===")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print("[OK] GPU Available")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Cores: {torch.cuda.get_device_capability(0)}")
else:
    print("[ERR] No GPU detected. CPU training will be slow for large models.")
    print("   You may have the CPU-only version of PyTorch installed.")
