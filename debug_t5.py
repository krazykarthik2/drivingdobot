import os
import sys

print("=== PINPOINTING FlaxT5EncoderModel SEGFAULT ===")

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("t5-base")
print("[1] AutoConfig t5-base loaded.")

print("[2] Importing FlaxT5EncoderModel class...")
from transformers import FlaxT5EncoderModel
print("  Class imported.")

print("[3] Testing FlaxT5EncoderModule directly (without PyTorch weight loading)...")
from transformers.models.t5.modeling_flax_t5 import FlaxT5EncoderModule
mod = FlaxT5EncoderModule(cfg)
print("  FlaxT5EncoderModule instantiated cleanly!")

print("[4] Testing FlaxT5EncoderModel(cfg)...")
try:
    m = FlaxT5EncoderModel(cfg)
    print("  FlaxT5EncoderModel initialized successfully!")
except Exception as e:
    print(f"  Exception: {e}")
