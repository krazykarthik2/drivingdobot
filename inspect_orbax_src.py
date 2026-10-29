import inspect
import orbax.checkpoint.utils as u

print("Source of _reconstruct_from_keypath:")
try:
    print(inspect.getsource(u._reconstruct_from_keypath))
except Exception as e:
    print(e)

print("\nSource of deserialize_tree:")
try:
    print(inspect.getsource(u.deserialize_tree))
except Exception as e:
    print(e)
