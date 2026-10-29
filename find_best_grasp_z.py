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

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()

# Test grasp at different Z levels: 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90
for test_z in [0.96, 0.94, 0.92, 0.90]:
    obs = env.reset()
    bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()
    # Descend to test_z
    for _ in range(40):
        curr_eef = obs["robot0_eef_pos"]
        diff = (np.array([bowl_pos[0], bowl_pos[1], test_z]) - curr_eef) * 5.0
        diff = np.clip(diff, -0.65, 0.65)
        obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))
    # Clamp
    for _ in range(30):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    # Lift
    for _ in range(30):
        obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))
    
    b_lift = env.env.sim.data.body_xpos[bowl_id]
    lifted = (b_lift[2] - bowl_pos[2]) > 0.05
    print(f"Test Z={test_z:.2f} -> Gripper Qpos={obs['robot0_gripper_qpos']} -> Bowl lifted: {lifted} (Delta Z={b_lift[2]-bowl_pos[2]:.3f})")

env.close()
