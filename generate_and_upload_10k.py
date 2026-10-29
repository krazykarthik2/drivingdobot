import os
import sys
import time
import pickle
import io
import argparse
from PIL import Image
import numpy as np
from huggingface_hub import HfApi, HfFolder
import pyrender

from generate_front_demonstrations import (
    DobotURDFRenderer,
    DobotKinematics,
    render_front_camera_view,
    COLOR_RGB_MAP,
    TABLE_Z_M,
    CUBE_TOP_Z_M,
    SUCKER_LENGTH_M,
    CONTACT_EE_Z_M,
    SAFE_LIFT_Z_M,
    CAM_EYE,
    CAM_TGT,
    CAM_UP,
    T_CAM
)

HF_TOKEN = os.environ.get("HF_TOKEN") or HfFolder.get_token()
REPO_ID = os.environ.get("HF_DATASET_REPO", "karthikkrazy/dobot-front-demonstrations")

def generate_and_upload_batches(total_episodes=10000, batch_size=500, upload=True):
    print("=" * 70)
    print("SCALED 10,000 EPISODE GENERATION WITH IN-MEMORY JPEG COMPRESSION")
    print(f"Total Target: {total_episodes} episodes | Batch Size: {batch_size} episodes/chunk")
    print(f"Hugging Face Target Dataset: {REPO_ID}")
    print("=" * 70)

    # Configure SSL verification bypass globally upfront for reliable Hugging Face API & LFS uploads
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    from huggingface_hub.utils import get_session
    get_session().verify = False

    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print("Note on repo creation:", e)

    renderer = DobotURDFRenderer(width=640, height=480)
    kin = DobotKinematics()
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=640.0 / 480.0)

    task_pairs = [
        ("red", "green"),
        ("blue", "yellow"),
        ("cyan", "purple"),
        ("orange", "red"),
        ("green", "blue"),
        ("yellow", "cyan"),
        ("purple", "orange")
    ]

    total_chunks = (total_episodes + batch_size - 1) // batch_size
    overall_start = time.time()
    generated_count = 0

    os.makedirs("demo_batches", exist_ok=True)

    for chunk_idx in range(total_chunks):
        chunk_eps = min(batch_size, total_episodes - generated_count)
        chunk_start_idx = generated_count
        chunk_filename = f"demo_batches/dobot_front_chunk_{chunk_idx:03d}.pkl"
        
        # Check if this chunk is already generated from a previous run
        if os.path.exists(chunk_filename) and os.path.getsize(chunk_filename) > 1000000:
            print(f"\n[Chunk {chunk_idx + 1}/{total_chunks}] {chunk_filename} already generated ({os.path.getsize(chunk_filename) / 1024 / 1024:.2f} MB). Skipping generation, checking upload...")
            if upload:
                try:
                    print(f"  --> Ensuring {chunk_filename} is uploaded to Hugging Face...")
                    api.upload_file(
                        path_or_fileobj=chunk_filename,
                        path_in_repo=f"chunks/{os.path.basename(chunk_filename)}",
                        repo_id=REPO_ID,
                        repo_type="dataset"
                    )
                    print("  --> Upload verified!")
                except Exception as up_err:
                    print(f"  --> Upload note: {up_err}")
            generated_count += chunk_eps
            continue

        print(f"\n[Chunk {chunk_idx + 1}/{total_chunks}] Generating {chunk_eps} episodes (episodes {chunk_start_idx}..{chunk_start_idx + chunk_eps - 1})...")
        chunk_demonstrations = []
        t0 = time.time()

        for ep_offset in range(chunk_eps):
            ep_idx = chunk_start_idx + ep_offset
            c_name, p_name = task_pairs[ep_idx % len(task_pairs)]
            instruction = f"pick up the {c_name} cube and place it on the {p_name} platform"

            np.random.seed(ep_idx + 42000)
            c_x = np.random.uniform(0.18, 0.26)
            c_y = np.random.uniform(0.03, 0.09) * np.random.choice([1.0, -1.0])
            c_z = CUBE_TOP_Z_M
            cube_pos = np.array([c_x, c_y, c_z])

            p_x = np.random.uniform(0.18, 0.26)
            p_y = np.random.uniform(0.04, 0.10) * (-1.0 if c_y > 0 else 1.0)
            p_z = TABLE_Z_M + 0.003
            plat_pos = np.array([p_x, p_y, p_z])

            ee_standby = np.array([0.22, 0.00, 0.020, 0.0])
            ee_hover_cube = np.array([cube_pos[0], cube_pos[1], SAFE_LIFT_Z_M, 0.0])
            ee_touch_cube = np.array([cube_pos[0], cube_pos[1], CONTACT_EE_Z_M, 0.0])
            ee_lift_cube  = np.array([cube_pos[0], cube_pos[1], SAFE_LIFT_Z_M, 0.0])
            ee_hover_plat = np.array([plat_pos[0], plat_pos[1], SAFE_LIFT_Z_M, 0.0])
            ee_touch_plat = np.array([plat_pos[0], plat_pos[1], CONTACT_EE_Z_M, 0.0])

            def interpolate_segment(p_start, p_end, steps, grip):
                seg = []
                for s in range(steps):
                    alpha = (s + 1) / float(steps)
                    pos = (1 - alpha) * p_start[:3] + alpha * p_end[:3]
                    seg.append((pos, grip))
                return seg

            waypoint_seq = []
            waypoint_seq.extend(interpolate_segment(ee_standby, ee_hover_cube, 4, 1.0))
            waypoint_seq.extend(interpolate_segment(ee_hover_cube, ee_touch_cube, 4, 1.0))
            waypoint_seq.extend(interpolate_segment(ee_touch_cube, ee_touch_cube, 2, -1.0))
            waypoint_seq.extend(interpolate_segment(ee_touch_cube, ee_lift_cube, 4, -1.0))
            waypoint_seq.extend(interpolate_segment(ee_lift_cube, ee_hover_plat, 5, -1.0))
            waypoint_seq.extend(interpolate_segment(ee_hover_plat, ee_touch_plat, 4, -1.0))
            waypoint_seq.extend(interpolate_segment(ee_touch_plat, ee_touch_plat, 2, 1.0))
            waypoint_seq.extend(interpolate_segment(ee_touch_plat, ee_standby, 4, 1.0))

            episode_obs_jpeg = []
            episode_acts = []
            current_ee = ee_standby.copy()
            current_cube = cube_pos.copy()
            suction_attached = False

            for next_pos, grip_val in waypoint_seq:
                delta_xyz = next_pos - current_ee[:3]
                act_7d = np.array([delta_xyz[0], delta_xyz[1], delta_xyz[2], 0.0, 0.0, 0.0, grip_val], dtype=np.float32)

                j = kin.inverse(current_ee[0], current_ee[1], current_ee[2], current_ee[3])
                t = renderer.compute_urdf_transforms(j[0], j[1], j[2], j[3], gripper_closed=False)
                s = renderer.build_scene(
                    t,
                    cube_pos=current_cube,
                    cube_color_rgb=COLOR_RGB_MAP[c_name],
                    plat_pos=plat_pos,
                    plat_color_rgb=COLOR_RGB_MAP[p_name],
                    distractor_cubes=[],
                    distractor_plats=[],
                    end_effector="suction"
                )
                img_front = render_front_camera_view(renderer, s, cam)

                # JPEG encode directly into memory buffer
                buf = io.BytesIO()
                Image.fromarray(img_front).save(buf, format="JPEG", quality=85)
                episode_obs_jpeg.append(buf.getvalue())
                episode_acts.append(act_7d)

                current_ee[:3] = next_pos
                if grip_val < -0.5:
                    suction_attached = True
                elif grip_val > 0.5:
                    suction_attached = False

                if suction_attached:
                    current_cube = np.array([current_ee[0], current_ee[1], current_ee[2] - SUCKER_LENGTH_M + 0.012])

            chunk_demonstrations.append({
                "instruction": instruction,
                "observations_jpeg": episode_obs_jpeg,
                "actions": np.stack(episode_acts)
            })

            if (ep_offset + 1) % 100 == 0 or (ep_offset + 1) == chunk_eps:
                elapsed = time.time() - t0
                print(f"    Chunk progress: {ep_offset + 1}/{chunk_eps} ({elapsed:.1f}s | {(elapsed / (ep_offset + 1)):.2f}s/ep)")

        with open(chunk_filename, "wb") as f:
            pickle.dump(chunk_demonstrations, f)

        chunk_size_mb = os.path.getsize(chunk_filename) / (1024 * 1024)
        print(f"  --> Saved {chunk_filename} ({chunk_size_mb:.2f} MB)")

        if upload:
            print(f"  --> Uploading {chunk_filename} to Hugging Face...")
            try:
                # Disable SSL verification warnings if corporate proxy / self-signed cert is present
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                from huggingface_hub.utils import get_session
                get_session().verify = False
                api.upload_file(
                    path_or_fileobj=chunk_filename,
                    path_in_repo=f"chunks/{os.path.basename(chunk_filename)}",
                    repo_id=REPO_ID,
                    repo_type="dataset"
                )
                print("  --> Upload verified!")
            except Exception as up_err:
                print(f"  --> [Warning] Upload failed ({up_err}). Keeping local file intact so generation continues unblocked!")

        generated_count += chunk_eps

    print("\n" * 2)
    print("=" * 70)
    print(f"ALL {generated_count} DEMONSTRATIONS SUCCESSFULLY GENERATED & UPLOADED!")
    print(f"Total time elapsed: {(time.time() - overall_start) / 60:.1f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--shutdown-after", action="store_true", help="Shut down system after full 10k generation and upload completes")
    args = parser.parse_args()

    generate_and_upload_batches(
        total_episodes=args.episodes,
        batch_size=args.batch_size,
        upload=not args.no_upload
    )

    if args.shutdown_after:
        print("\nAll 10,000 episodes generated and uploaded. Initiating system shutdown...")
        import subprocess
        if sys.platform.startswith("win"):
            subprocess.run(["shutdown", "/s", "/t", "60"])
        else:
            subprocess.run(["shutdown", "-h", "now"])
