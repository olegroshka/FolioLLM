import torch
from transformers import TrainerCallback

class MemoryMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    def on_epoch_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print("Cleared CUDA cache at the start of the epoch.")
        print("\nCleared CUDA cache at the start of the epoch.")
