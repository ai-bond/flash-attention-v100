__version__ = "26.06"

from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_gpu,
    flash_attn_varlen_func,
    flash_attn_varlen_gpu,
    flash_attn_with_kvcache,
    flash_attn_with_kvcache_gpu
)

from .cross_entropy import cross_entropy_loss

__all__ = [
    "flash_attn_func",
    "flash_attn_gpu",
    "flash_attn_varlen_func",
    "flash_attn_varlen_gpu",
    "flash_attn_with_kvcache",
    "flash_attn_with_kvcache_gpu",
    "cross_entropy_loss",
]