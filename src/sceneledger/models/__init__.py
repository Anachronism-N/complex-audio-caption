"""Optional PyTorch model components. Import submodules only when torch is installed."""
from .decode import decode_slot_arrays

__all__ = ["decode_slot_arrays"]
