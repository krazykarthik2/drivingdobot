"""
Zero-Shot Evaluation of Pretrained Official 3D Diffusion Policy (DP3) on LIBERO.
Takes official weights trained on 3D manipulation tasks (MetaWorld bin-picking),
projects LIBERO RGB-D into 512-point surface clouds, conditions DP3 on the 3D geometry,
and executes the predicted 3D action trajectory on Franka Panda.
"""
import os
import sys
import datetime
import numpy as np
import torch
from omegaconf import OmegaConf

# Add DP3 repo to path
sys.path.insert(0, os.path.abspath("dp3_official/3D-Diffusion-Policy"))
from diffusion_policy_3d.policy.dp3 import DP3
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import robosuite.utils.transform_utils as T
from libero.libero import benchmark

def load_pretrained_dp3(ckpt_path="dp3_official/latest_binpicking.ckpt"):
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    OmegaConf.register_new_resolver('now', lambda fmt: datetime.datetime.now().strftime(fmt), replace=True)

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    resolved_cfg = OmegaConf.to_container(ckpt["cfg"], resolve=True)
    p_cfg = resolved_cfg["policy"]

    scheduler_args = dict(p_cfg["noise_scheduler"])
    scheduler_args.pop("_target_", None)
    noise_scheduler = DDIMScheduler(**scheduler_args)

    policy = DP3(
        shape_meta=resolved_cfg["shape_meta"],
        noise_scheduler=noise_scheduler,
        horizon=resolved_cfg["horizon"],
        n_action_steps=resolved_cfg["n_action_steps"],
        n_obs_steps=resolved_cfg["n_obs_steps"],
        num_inference_steps=p_cfg["num_inference_steps"],
        obs_as_global_cond=p_cfg["obs_as_global_cond"],
        diffusion_step_embed_dim=p_cfg["diffusion_step_embed_dim"],
        down_dims=p_cfg["down_dims"],
        kernel_size=p_cfg["kernel_size"],
        n_groups=p_cfg["n_groups"],
        condition_type=p_cfg["condition_type"],
        use_down_condition=p_cfg["use_down_condition"],
        use_mid_condition=p_cfg["use_mid_condition"],
        use_up_condition=p_cfg["use_up_condition"],
        encoder_output_dim=p_cfg["encoder_output_dim"],
        crop_shape=p_cfg["crop_shape"],
        use_pc_color=p_cfg["use_pc_color"],
        pointnet_type=p_cfg["pointnet_type"],
        pointcloud_encoder_cfg=OmegaConf.create(p_cfg["pointcloud_encoder_cfg"])
    )

    policy.load_state_dict(ckpt["state_dicts"]["model"])
    policy.eval()
    print("Pretrained DP3 policy successfully loaded into CPU memory!")
    return policy, resolved_cfg

def extract_libero_point_cloud(obs, env, num_points=512):
    """
    Extract 3D point cloud from LIBERO observation using camera depth & extrinsics.
    """
    # Try agentview depth if available
    depth = obs.get("agentview_depth", None)
    if depth is None:
        # Fall back to joint/EE point cloud sampling around workspace
        ee_pos = obs.get("robot0_eef_pos", np.zeros(3))
        # Generate workspace bounding box points centered at robot EEF
        noise = np.random.uniform(-0.15, 0.15, size=(num_points, 3)).astype(np.float32)
        pts = ee_pos.astype(np.float32) + noise
        return pts

    # Squeeze depth map
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    # In MuJoCo / Robosuite, depth is normalized [0, 1] relative to near and far clipping planes
    sim = env.sim
    cam_id = sim.model.camera_name2id("agentview")
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    depth_m = near / (1.0 - depth * (1.0 - near / far) + 1e-8)

    H, W = depth.shape
    fov = sim.model.cam_fovy[cam_id]
    f = 0.5 * H / np.tan(np.deg2rad(fov) / 2.0)
    cx, cy = W / 2.0, H / 2.0

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth_m > 0.1) & (depth_m < 2.0)
    z = depth_m[valid]
    x = (u[valid] - cx) * z / f
    y = (v[valid] - cy) * z / f
    pts_cam = np.stack([x, -y, -z], axis=-1)

    # Transform to robot base / world coordinates using camera pose
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    pts_world = (cam_mat @ pts_cam.T).T + cam_pos

    # Downsample / sample to 512 points
    N = len(pts_world)
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
        return pts_world[idx].astype(np.float32)
    elif N > 0:
        idx = np.random.choice(N, num_points, replace=True)
        return pts_world[idx].astype(np.float32)
    else:
        ee_pos = obs.get("robot0_eef_pos", np.zeros(3))
        noise = np.random.uniform(-0.15, 0.15, size=(num_points, 3)).astype(np.float32)
        return ee_pos.astype(np.float32) + noise

def evaluate_task(policy, cfg, benchmark_dict, task_id=0, max_steps=200):
    task = benchmark_dict.get_task(task_id)
    task_name = task.name
    task_desc = task.language
    print(f"\n==========================================")
    print(f"Zero-Shot Evaluation: Task {task_id}: {task_name}")
    print(f"Instruction: {task_desc}")
    print(f"==========================================")

    init_states = benchmark_dict.get_task_init_states(task_id)
    env_args = {
        "bddl_file_name": benchmark_dict.get_task_bddl_file_path(task_id),
        "camera_heights": 128,
        "camera_widths": 128,
        "camera_depths": True,
    }
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(**env_args)
    env.seed(42)
    env.reset()
    env.set_init_state(init_states[0])

    # Warmup
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    n_obs_steps = cfg["n_obs_steps"]
    history_pc = []
    history_agent_pos = []

    success = False
    for step in range(max_steps):
        pc = extract_libero_point_cloud(obs, env, num_points=512)
        ee_pos = obs.get("robot0_eef_pos", np.zeros(3))
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
        # MetaWorld agent_pos is shape [9]: ee_pos(3) + gripper(2) + dummy/quat(4)
        agent_pos = np.concatenate([ee_pos, gripper_qpos, np.zeros(4)]).astype(np.float32)

        history_pc.append(pc)
        history_agent_pos.append(agent_pos)

        if len(history_pc) < n_obs_steps:
            obs, _, _, _ = env.step(np.zeros(7))
            continue

        cur_pcs = np.stack(history_pc[-n_obs_steps:], axis=0) # (2, 512, 3)
        cur_agents = np.stack(history_agent_pos[-n_obs_steps:], axis=0) # (2, 9)

        batch_obs = {
            "point_cloud": torch.from_numpy(cur_pcs).unsqueeze(0).float(),
            "agent_pos": torch.from_numpy(cur_agents).unsqueeze(0).float()
        }

        with torch.no_grad():
            action_dict = policy.predict_action(batch_obs)
        pred_actions = action_dict["action"][0].cpu().numpy() # (3, 4): [dx, dy, dz, gripper]

        # Execute the first predicted action step
        dx, dy, dz, grip = pred_actions[0]
        # Clip delta action to safe operational bounds
        dp = np.clip([dx, dy, dz], -0.05, 0.05)
        # LIBERO Franka action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        franka_action = np.array([dp[0], dp[1], dp[2], 0.0, 0.0, 0.0, np.clip(grip, -1.0, 1.0)])

        obs, reward, done, info = env.step(franka_action)
        if env.check_success():
            print(f"Task {task_id} SUCCEEDED at step {step}!")
            success = True
            break

    if not success:
        print(f"Task {task_id} completed: Success={success} after {max_steps} steps.")
    env.close()
    return success

def main():
    policy, cfg = load_pretrained_dp3()
    bm = benchmark.get_benchmark_dict()
    spatial_suite = bm["libero_spatial"]()
    num_tasks = spatial_suite.get_num_tasks()
    print(f"LIBERO Spatial suite contains {num_tasks} tasks.")

    results = {}
    for tid in range(min(5, num_tasks)):
        res = evaluate_task(policy, cfg, spatial_suite, task_id=tid, max_steps=120)
        results[tid] = res

    print("\n" + "="*50)
    print("ZERO-SHOT PRETRAINED DP3 BENCHMARK RESULTS:")
    for tid, res in results.items():
        print(f"Task {tid}: {'PASSED' if res else 'FAILED'}")
    succ_rate = sum(results.values()) / len(results) * 100.0
    print(f"Success Rate: {succ_rate:.1f}% ({sum(results.values())}/{len(results)})")
    print("="*50)

if __name__ == "__main__":
    main()
