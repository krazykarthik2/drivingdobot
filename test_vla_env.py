import sys

def run_diagnostics():
    print("=" * 65)
    print("VM GPU ENVIRONMENT DIAGNOSTIC HARNESS")
    print("=" * 65)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]}")

    # 1. JAX & Accelerator Probe
    print("\n[1/5] Checking JAX & Accelerator...")
    try:
        import jax
        print(f"  JAX Version:       {jax.__version__}")
        print(f"  Default Backend:   {jax.default_backend().upper()}")
        print(f"  Available Devices: {jax.devices()}")
    except Exception as e:
        print(f"  [ERROR] JAX initialization failed: {e}")

    # 2. NumPy ABI Check
    print("\n[2/5] Checking NumPy Version & C-API ABI...")
    try:
        import numpy as np
        print(f"  NumPy Version:     {np.__version__}")
        if int(np.__version__.split('.')[0]) >= 2:
            print("  [WARNING] NumPy >= 2.0.0 detected! tensorstore requires numpy < 2.0.0.")
        else:
            print("  [OK] NumPy 1.x compatible ABI.")
    except Exception as e:
        print(f"  [ERROR] NumPy check failed: {e}")

    # 3. Transformers & Flax Integration
    print("\n[3/5] Checking HuggingFace Transformers & FlaxAutoModel...")
    try:
        import transformers
        print(f"  Transformers Ver:  {transformers.__version__}")
        print(f"  is_flax_available: {transformers.is_flax_available()}")
        try:
            from transformers import FlaxAutoModel, FlaxT5EncoderModel
            print("  [OK] Directly imported FlaxAutoModel & FlaxT5EncoderModel from transformers.")
        except Exception as e:
            print(f"  [MISSING] 'from transformers import FlaxAutoModel' failed: {e}")
            print("  Testing underlying module import...")
            try:
                import transformers.models.auto.modeling_flax_auto as m
                print(f"  [OK] Direct subpath import works: hasattr(m, 'FlaxAutoModel') -> {hasattr(m, 'FlaxAutoModel')}")
            except Exception as sub_e:
                print(f"  [ERROR] Direct subpath import also failed: {sub_e}")
    except Exception as e:
        print(f"  [ERROR] Transformers check failed: {e}")

    # 4. Octo & dlimp Probe
    print("\n[4/5] Checking Octo & dlimp...")
    try:
        import octo_patch
        import octo
        print(f"  Octo loaded from:  {octo.__file__}")
        print("  octo_patch loaded cleanly.")
    except Exception as e:
        print(f"  [ERROR] Octo import failed: {e}")

    # 5. Checkpoint & Pretrained Model Loading
    print("\n[5/5] Testing OctoModel.load_pretrained('octo_small_checkpoint')...")
    try:
        from octo.model.octo_model import OctoModel
        model = OctoModel.load_pretrained("octo_small_checkpoint")
        print("  [SUCCESS] OctoModel loaded completely without errors!")
        print(f"  Model parameter keys: {list(model.params.keys())}")
    except Exception as e:
        print(f"  [FAILED] Model loading failed: {e}")

    print("\n" + "=" * 65)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    run_diagnostics()
