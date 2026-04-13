from flash_attn_v100 import flash_attn_func, flash_attn_gpu, __all__

__version__ = "2.8.3"

__all__ = __all__ if '__all__' in locals() else ["flash_attn_func"]
__version__ = __version__ if '__version__' in locals() else "2.8.3"
__doc__ = f"Flash Attention for Tesla V100 v{__version__}"