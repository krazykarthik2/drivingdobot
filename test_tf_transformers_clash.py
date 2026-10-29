import sys
print("Checking TensorFlow before Transformers...")

import tensorflow as tf
print("1. TensorFlow imported")

print("2. Now importing transformers FlaxT5EncoderModel...")
from transformers import FlaxT5EncoderModel
print("3. FlaxT5EncoderModel imported after TensorFlow!")
