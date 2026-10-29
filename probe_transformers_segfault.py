import sys
print("Testing transformers sub-imports...")

print("[A] import transformers (root)")
import transformers
print("  root ok")

print("[B] import AutoConfig")
from transformers import AutoConfig
print("  AutoConfig ok")

print("[C] importing FlaxT5EncoderModel directly from modeling_flax_t5...")
import transformers.models.t5.modeling_flax_t5 as t5
print("  direct subpath ok:", hasattr(t5, "FlaxT5EncoderModel"))

print("[D] from transformers import FlaxT5EncoderModel")
from transformers import FlaxT5EncoderModel
print("  symbol ok")
