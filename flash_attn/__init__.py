from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from flash_attn_v100 import (
    flash_attn_func,
    flash_attn_gpu,
    flash_attn_varlen_func,
    flash_attn_varlen_gpu,
    __version__ as _v100_version
)

__version__ = "2.8.3"

__all__ = [
    "flash_attn_func",
    "flash_attn_gpu",
    "flash_attn_varlen_func",
    "flash_attn_varlen_gpu",
    "__version__"
]

__doc__ = f"Flash Attention for Tesla V100 v{__version__} (backend: v{_v100_version})"