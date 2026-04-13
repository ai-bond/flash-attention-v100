from flash_attn_v100 import flash_attn_func, flash_attn_gpu

flash_attn_gpu = flash_attn_gpu
flash_attn_func = flash_attn_func

__all__ = ["flash_attn_gpu", "flash_attn_func"]
