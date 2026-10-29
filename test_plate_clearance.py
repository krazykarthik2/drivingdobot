import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_spatial")()
task = b.get_task(0)
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_widths=128, camera_heights=128, render_gpu_device_id=-1)
obs = env.reset()

plate_id = env.env.obj_body_id["plate_1"]
plate_init = env.env.sim.data.body_xpos[plate_id].copy()

# Move at Z=0.940, Z=0.960, Z=0.980, Z=1.000 over plate center
for test_z in [0.940, 0.960, 0.980, 1.000]:
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((np.array([plate_init[0], plate_init[1], test_z]) - curr) * 10.0, -0.6, 0.6)
    obs, _, _, _ = env.step(action)
    plate_curr = env.env.sim.data.body_xpos[plate_id]
    delta = np.linalg.norm(plate_curr[:2] - plate_init[:2])
    print(f"Test Z={test_z:.3f}: Plate displaced by {delta*100:.2f} cm")

env.close()
