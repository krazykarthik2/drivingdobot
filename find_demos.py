import os
import h5py
import libero

libero_path = os.path.dirname(libero.__file__)
demo_path = os.path.join(libero_path, "libero", "datasets")
print("Looking for demo datasets in:", demo_path)
found_h5 = []
for root, dirs, files in os.walk(os.path.expanduser("~")):
    for f in files:
        if f.endswith(".hdf5") and "spatial" in f:
            found_h5.append(os.path.join(root, f))
    if len(found_h5) > 0:
        break

print(f"Found {len(found_h5)} hdf5 demo files:")
for f in found_h5[:3]:
    print(" ", f)
