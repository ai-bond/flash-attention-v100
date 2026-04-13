__version__ = "26.04"

from .flash_attn_interface import flash_attn_func, flash_attn_gpu

__all__ = ["flash_attn_func", "flash_attn_gpu"]
