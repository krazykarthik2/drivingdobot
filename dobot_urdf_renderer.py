import os
import sys

# Configure headless offscreen rendering for pyrender on Linux
# In headless Linux containers without DISPLAY or sudo for xvfb, surfaceless EGL is used.
if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_PLATFORM"] = "surfaceless"

try:
    import pyrender.platforms.egl as _pyrender_egl
    # In headless containers, query_devices() fails or returns [] because of hypervisor restrictions.
    # Force get_device_by_index and get_default_device to return EGLDevice(None) directly,
    # which binds egl.EGL_DEFAULT_DISPLAY in surfaceless mode without crashing.
    def _safe_get_device(device_id=0):
        return _pyrender_egl.EGLDevice(None)
    _pyrender_egl.get_device_by_index = _safe_get_device
    _pyrender_egl.get_default_device = _safe_get_device
except Exception:
    pass

import numpy as np
import trimesh
import pyrender
import cv2
import PIL.Image

DOBOT_MESH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "dobot_description", "meshes", "dae"))

class DobotURDFRenderer:
    """
    Physically accurate 3D scene and robot renderer using Dobot Magician URDF geometry and collada meshes.
    Supports forward kinematics, overhead observation camera, wrist camera,
    and third-person cinematic rendering with high-resolution video production.
    """
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        
        # Load raw meshes once
        self.raw_meshes = {
            "base": trimesh.load(os.path.join(DOBOT_MESH_DIR, "magicianBase.dae"), force="mesh"),
            "link1": trimesh.load(os.path.join(DOBOT_MESH_DIR, "magicianLink1.dae"), force="mesh"),
            "link2": trimesh.load(os.path.join(DOBOT_MESH_DIR, "magicianLink2.dae"), force="mesh"),
            "link3": trimesh.load(os.path.join(DOBOT_MESH_DIR, "magicianLink3.dae"), force="mesh"),
            "link4": trimesh.load(os.path.join(DOBOT_MESH_DIR, "magicianLink4.dae"), force="mesh"),
            "gripper": trimesh.load(os.path.join(DOBOT_MESH_DIR, "gripper_core.dae"), force="mesh"),
            "jaw_l": trimesh.load(os.path.join(DOBOT_MESH_DIR, "jaw_left.dae"), force="mesh"),
            "jaw_r": trimesh.load(os.path.join(DOBOT_MESH_DIR, "jaw_right.dae"), force="mesh"),
        }

        # Materials
        self.mat_robot_silver = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.88, 0.90, 0.93, 1.0], metallicFactor=0.4, roughnessFactor=0.35
        )
        self.mat_robot_black = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.20, 0.22, 0.26, 1.0], metallicFactor=0.2, roughnessFactor=0.5
        )

        # Single offscreen renderer to maintain one OpenGL context
        self.r = pyrender.OffscreenRenderer(width, height)

    @staticmethod
    def make_transform(xyz, rpy):
        cr, cp, cy = np.cos(rpy)
        sr, sp, sy = np.sin(rpy)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = xyz
        return T

    def compute_urdf_transforms(self, j1, j2_kin, j3_kin, j4, gripper_closed=False):
        """
        Computes forward kinematics for all links matching the URDF definition exactly:
        URDF joint 1 = j1
        URDF joint 2 = pi/2 - j2_kin
        URDF joint 3 = -pi/2 - j3_kin
        URDF joint mimic 1 = -j2_urdf
        URDF joint mimic 2 = -j3_urdf
        URDF joint 4 = j4
        """
        j2_urdf = np.pi/2 - j2_kin
        j3_urdf = -np.pi/2 - j3_kin

        # Transform stack
        T_root_base = self.make_transform([0, 0, 0.138], [0, 0, 0])
        T_base_l1 = self.make_transform([0, 0, 0], [0, 0, j1])
        T_l1_l2 = self.make_transform([0, 0, 0], [0, j2_urdf, 0])
        T_l2_l3 = self.make_transform([0, 0, 0.135], [0, j3_urdf, 0])
        T_l3_v = self.make_transform([0.147, 0, 0], [0, -j2_urdf, 0])
        T_v_l4 = self.make_transform([0, 0, 0], [0, -j3_urdf, 0])
        T_l4_core = self.make_transform([0.06, 0, 0], [0, 0, j4])

        # World transformations for links
        W_base = T_root_base
        W_l1 = W_base @ T_base_l1
        W_l2 = W_l1 @ T_l1_l2
        W_l3 = W_l2 @ T_l2_l3
        W_l4 = W_l3 @ T_l3_v @ T_v_l4
        W_gripper = W_l4 @ T_l4_core

        # Visual origin offsets from URDF
        V_l2 = W_l2 @ self.make_transform([0, 0, 0], [0, -0.34906585, 0])
        V_l3 = W_l3 @ self.make_transform([0.01645236, 0, -0.1339937], [0, -0.4712389, 0])
        V_l4 = W_l4 @ self.make_transform([-0.17715068, 0, -0.0585951], [0, 0, 0])
        V_gripper = W_gripper @ self.make_transform([-8.0678e-05, 0, -0.0278434], [0, 0, 0])

        # Jaws
        jaw_disp = -0.008 if gripper_closed else 0.003
        V_jaw_l = W_gripper @ self.make_transform([-8.0678e-05, -0.0009 + jaw_disp, -0.0278434], [0, 0, 0])
        V_jaw_r = W_gripper @ self.make_transform([-8.0678e-05, 0.0009 - jaw_disp, -0.0278434], [0, 0, 0])

        return {
            "base": W_base,
            "link1": W_l1,
            "link2": V_l2,
            "link3": V_l3,
            "link4": V_l4,
            "gripper": V_gripper,
            "jaw_l": V_jaw_l,
            "jaw_r": V_jaw_r,
            "ee_world": W_gripper,
            "ee_pos": W_gripper[:3, 3]
        }

    def _create_pyrender_meshes(self):
        """Creates freshly allocated pyrender meshes for new scene."""
        return {
            "base": pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=self.raw_meshes["base"].vertices, faces=self.raw_meshes["base"].faces), material=self.mat_robot_black, smooth=True),
            "link1": pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=self.raw_meshes["link1"].vertices, faces=self.raw_meshes["link1"].faces), material=self.mat_robot_silver, smooth=True),
            "link2": pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=self.raw_meshes["link2"].vertices, faces=self.raw_meshes["link2"].faces), material=self.mat_robot_silver, smooth=True),
            "link3": pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=self.raw_meshes["link3"].vertices, faces=self.raw_meshes["link3"].faces), material=self.mat_robot_silver, smooth=True),
            "link4": pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=self.raw_meshes["link4"].vertices, faces=self.raw_meshes["link4"].faces), material=self.mat_robot_black, smooth=True),
            "gripper": pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=self.raw_meshes["gripper"].vertices, faces=self.raw_meshes["gripper"].faces), material=self.mat_robot_black, smooth=True),
            "jaw_l": pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=self.raw_meshes["jaw_l"].vertices, faces=self.raw_meshes["jaw_l"].faces), material=self.mat_robot_silver, smooth=True),
            "jaw_r": pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=self.raw_meshes["jaw_r"].vertices, faces=self.raw_meshes["jaw_r"].faces), material=self.mat_robot_silver, smooth=True),
        }

    def build_scene(self, transforms, cube_pos=None, cube_color_rgb=None, plat_pos=None, plat_color_rgb=None,
                    distractor_cubes=None, distractor_plats=None, end_effector="suction",
                    table_texture=None, workspace_size_m=0.36):
        # Vivid ambient light so table projection and robot geometry are crisp and bright
        scene = pyrender.Scene(bg_color=[0.08, 0.09, 0.12, 1.0], ambient_light=[0.72, 0.72, 0.72])

        # 1. Physical Table Surface
        # Ground-truth table plane from physical Dobot measurement:
        # With sucker resting on table, hardware measures: EE Z = -69.2 mm (-0.0692 m).
        # Sucker extends 80mm downwards -> Sucker tip touches table at Z = -0.0692 - 0.080 = -0.1492 m.
        table_top_z = -0.1492
        table_thick = 0.020
        table_mesh = trimesh.creation.box(extents=[0.58, 0.58, table_thick])
        mat_table = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.14, 0.16, 0.20, 1.0], roughnessFactor=0.9)
        table_node = pyrender.Mesh.from_trimesh(trimesh.Trimesh(table_mesh.vertices, table_mesh.faces), material=mat_table, smooth=False)
        scene.add(table_node, pose=self.make_transform([0.22, 0.0, table_top_z - table_thick/2.0], [0, 0, 0]))

        # Project Live Calibrated Camera POV directly onto the workspace table surface
        if table_texture is not None:
            try:
                # 64x64 vertex color grid for high resolution, vibrant real-time rendering
                grid_size = 64
                tex_rgb = cv2.cvtColor(table_texture, cv2.COLOR_BGR2RGB) if len(table_texture.shape) == 3 else table_texture
                sampled = cv2.resize(tex_rgb, (grid_size, grid_size), interpolation=cv2.INTER_AREA)

                # Physical table mapping matching camera perspective:
                # - Screen Left (u=0) is Robot Right (-Y = -140mm)
                # - Screen Right (u=w-1) is Robot Left (+Y = +140mm)
                # - Screen Top (v=0) is near Dobot base (+X = 80mm)
                # - Screen Bottom (v=h-1) is far table edge (+X = 360mm)
                # Local table mesh is centered at X = 0.220 m (220 mm), span = 0.280 m (-0.140 to +0.140 m)
                half_s = 0.140 # 140 mm
                xs = np.linspace(-half_s, half_s, grid_size) # from base (-half_s -> X=80mm) to forward (+half_s -> X=360mm)
                ys = np.linspace(-half_s, half_s, grid_size) # from right (-half_s -> Y=-140mm) to left (+half_s -> Y=+140mm)
                xx, yy = np.meshgrid(xs, ys, indexing='ij')

                verts = np.column_stack([xx.flatten(), yy.flatten(), np.zeros(grid_size * grid_size, dtype=np.float32)])

                colors = np.zeros((grid_size * grid_size, 4), dtype=np.uint8)
                colors[:, :3] = sampled.reshape(-1, 3)
                colors[:, 3] = 255

                # Construct faces with normals pointing strictly UPWARDS (+Z) towards camera and lights
                faces = []
                for i in range(grid_size - 1):
                    for j in range(grid_size - 1):
                        idx = i * grid_size + j
                        faces.append([idx, idx + grid_size, idx + 1])
                        faces.append([idx + 1, idx + grid_size, idx + grid_size + 1])
                faces = np.array(faces, dtype=np.int32)

                proj_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors)
                p_proj_node = pyrender.Mesh.from_trimesh(proj_mesh, smooth=False)
                scene.add(p_proj_node, pose=self.make_transform([0.22, 0.0, table_top_z + 0.001], [0, 0, 0]))
            except Exception as e:
                border = trimesh.creation.box(extents=[0.52, 0.52, 0.002])
                mat_border = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.24, 0.26, 0.32, 1.0], roughnessFactor=0.9)
                scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(border.vertices, border.faces), material=mat_border, smooth=False), pose=self.make_transform([0.22, 0.0, table_top_z + 0.001], [0, 0, 0]))
        else:
            border = trimesh.creation.box(extents=[0.52, 0.52, 0.002])
            mat_border = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.24, 0.26, 0.32, 1.0], roughnessFactor=0.9)
            scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(border.vertices, border.faces), material=mat_border, smooth=False), pose=self.make_transform([0.22, 0.0, table_top_z + 0.001], [0, 0, 0]))

        # 2. Dobot Links
        p_meshes = self._create_pyrender_meshes()
        # Add base and arm links 1-4
        for name in ["base", "link1", "link2", "link3", "link4"]:
            scene.add(p_meshes[name], pose=transforms[name])

        if end_effector == "suction":
            # Suction Cup: Extends exactly 8.0 cm (0.080 m) downwards from end-effector core
            # Stem (cylinder)
            stem = trimesh.creation.cylinder(radius=0.008, height=0.065)
            mat_stem = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.15, 0.15, 0.17, 1.0], metallicFactor=0.8)
            p_stem = pyrender.Mesh.from_trimesh(trimesh.Trimesh(stem.vertices, stem.faces), material=mat_stem, smooth=True)
            # Suction cup rubber head
            cup = trimesh.creation.cylinder(radius=0.014, height=0.015)
            mat_cup = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.9, 0.35, 0.1, 1.0], roughnessFactor=0.9)
            p_cup = pyrender.Mesh.from_trimesh(trimesh.Trimesh(cup.vertices, cup.faces), material=mat_cup, smooth=True)

            ee_pose = transforms["ee_world"]
            T_stem = ee_pose @ self.make_transform([0, 0, -0.035], [0, 0, 0])
            T_cup = ee_pose @ self.make_transform([0, 0, -0.0725], [0, 0, 0])
            scene.add(p_stem, pose=T_stem)
            scene.add(p_cup, pose=T_cup)
        else:
            # Gripper with jaws
            for name in ["gripper", "jaw_l", "jaw_r"]:
                scene.add(p_meshes[name], pose=transforms[name])

        # 3. Target Platform
        if plat_pos is not None:
            p_rgb = plat_color_rgb if plat_color_rgb else [0.2, 0.85, 0.3]
            plat_mesh = trimesh.creation.cylinder(radius=0.035, height=0.006)
            mat_plat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*p_rgb, 1.0], roughnessFactor=0.4)
            p_plat = pyrender.Mesh.from_trimesh(trimesh.Trimesh(plat_mesh.vertices, plat_mesh.faces), material=mat_plat, smooth=True)
            scene.add(p_plat, pose=self.make_transform(plat_pos, [0, 0, 0]))

        # 4. Target Cube
        if cube_pos is not None:
            c_rgb = cube_color_rgb if cube_color_rgb else [0.9, 0.2, 0.2]
            cube_mesh = trimesh.creation.box(extents=[0.024, 0.024, 0.024])
            mat_cube = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*c_rgb, 1.0], roughnessFactor=0.3)
            p_cube = pyrender.Mesh.from_trimesh(trimesh.Trimesh(cube_mesh.vertices, cube_mesh.faces), material=mat_cube, smooth=False)
            scene.add(p_cube, pose=self.make_transform(cube_pos, [0, 0, 0]))

        # 5. Distractors
        if distractor_cubes:
            for d_rgb, d_pos in distractor_cubes:
                dc_m = trimesh.creation.box(extents=[0.022, 0.022, 0.022])
                mat_dc = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*d_rgb, 1.0], roughnessFactor=0.3)
                p_dc = pyrender.Mesh.from_trimesh(trimesh.Trimesh(dc_m.vertices, dc_m.faces), material=mat_dc, smooth=False)
                scene.add(p_dc, pose=self.make_transform(d_pos, [0, 0, 0]))

        if distractor_plats:
            for dp_rgb, dp_pos in distractor_plats:
                dp_m = trimesh.creation.cylinder(radius=0.035, height=0.006)
                mat_dp = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*dp_rgb, 1.0], roughnessFactor=0.4)
                p_dp = pyrender.Mesh.from_trimesh(trimesh.Trimesh(dp_m.vertices, dp_m.faces), material=mat_dp, smooth=True)
                scene.add(p_dp, pose=self.make_transform(dp_pos, [0, 0, 0]))

        return scene

    def render_views(self, scene, ee_pos):
        """
        Renders from the single renderer context:
        - Primary camera: overhead view (resized to 256x256)
        - Wrist camera: end-effector POV (resized to 128x128)
        - Third-person camera: 640x480
        """
        # --- 1. Third Person View (640x480) ---
        # Canonical Bridge-style shoulder/quarter viewpoint: looking from base shoulder forward into workspace
        # Eye positioned at [-0.05, -0.18, 0.32], looking at [0.22, 0.02, -0.05]
        cam_tp = pyrender.PerspectiveCamera(yfov=np.radians(52.0))
        eye_tp = np.array([-0.05, -0.18, 0.32])
        tgt_tp = np.array([0.22, 0.02, -0.05])
        up_tp = np.array([0, 0, 1])
        fwd_tp = (tgt_tp - eye_tp) / np.linalg.norm(tgt_tp - eye_tp)
        rgt_tp = np.cross(fwd_tp, up_tp) / np.linalg.norm(np.cross(fwd_tp, up_tp))
        true_up_tp = np.cross(rgt_tp, fwd_tp)

        T_tp = np.eye(4)
        T_tp[:3, :3] = np.column_stack([rgt_tp, true_up_tp, -fwd_tp])
        T_tp[:3, 3] = eye_tp

        cam_tp_node = scene.add(cam_tp, pose=T_tp)
        light_tp_node = scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5), pose=T_tp)

        img_tp, _ = self.r.render(scene)
        scene.remove_node(cam_tp_node)
        scene.remove_node(light_tp_node)

        # --- 2. Primary Camera (Overhead view for Octo) ---
        cam_primary = pyrender.PerspectiveCamera(yfov=np.radians(52.0))
        eye_p = np.array([0.22, -0.01, 0.42])
        tgt_p = np.array([0.22, 0.0, 0.01])
        up_p = np.array([1, 0, 0])
        fwd_p = (tgt_p - eye_p) / np.linalg.norm(tgt_p - eye_p)
        rgt_p = np.cross(fwd_p, up_p) / np.linalg.norm(np.cross(fwd_p, up_p))
        true_up_p = np.cross(rgt_p, fwd_p)

        T_primary = np.eye(4)
        T_primary[:3, :3] = np.column_stack([rgt_p, true_up_p, -fwd_p])
        T_primary[:3, 3] = eye_p

        cam_p_node = scene.add(cam_primary, pose=T_primary)
        light_p_node = scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5), pose=T_primary)

        img_p_raw, _ = self.r.render(scene)
        scene.remove_node(cam_p_node)
        scene.remove_node(light_p_node)

        # Resize to 256x256
        img_primary = cv2.resize(img_p_raw, (256, 256), interpolation=cv2.INTER_AREA)

        # --- 3. Wrist Camera (End-Effector POV for Octo) ---
        cam_wrist = pyrender.PerspectiveCamera(yfov=np.radians(65.0))
        eye_w = np.array([ee_pos[0], ee_pos[1], max(ee_pos[2], 0.08)])
        tgt_w = np.array([ee_pos[0], ee_pos[1], 0.0])
        up_w = np.array([1, 0, 0])
        fwd_w = (tgt_w - eye_w) / np.linalg.norm(tgt_w - eye_w)
        rgt_w = np.cross(fwd_w, up_w) / np.linalg.norm(np.cross(fwd_w, up_w))
        true_up_w = np.cross(rgt_w, fwd_w)

        T_wrist = np.eye(4)
        T_wrist[:3, :3] = np.column_stack([rgt_w, true_up_w, -fwd_w])
        T_wrist[:3, 3] = eye_w

        cam_w_node = scene.add(cam_wrist, pose=T_wrist)
        light_w_node = scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=T_wrist)

        img_w_raw, _ = self.r.render(scene)
        scene.remove_node(cam_w_node)
        scene.remove_node(light_w_node)

        # Resize to 128x128
        img_wrist = cv2.resize(img_w_raw, (128, 128), interpolation=cv2.INTER_AREA)

        return img_primary, img_wrist, img_tp

    def close(self):
        self.r.delete()
