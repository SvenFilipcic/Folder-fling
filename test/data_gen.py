"""
data_gen.py  —  Generate training data for UniGarmentManip retraining

For each sample:
  - Random orientation drop → settle → raycast occlusion → FPS 2048
  - Save .npz with mesh_points (full), pcd_points (2048), visible_mesh_indices

Output structure (matches dataloader_only_cd.py expectations):
  data/majca/
      0_majca/
          majca_0000.npz
          majca_0001.npz
          ...

Run:
    PYTHON_PATH test/data_gen.py
    PYTHON_PATH test/data_gen.py --samples 2000
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--outdir",  type=str, default=None, help="Override output directory")
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
carb.settings.get_settings().set_bool("/physics/suppressReadback", True)

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim, ClothPrim, ParticleSystem
from omni.isaac.core.materials.particle_material import ParticleMaterial
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom
from Env.Utils.pointcloud import furthest_point_sampling

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DRESS_USD    = os.path.join(PROJECT_ROOT, "assets/garments/obleka2/majca_hires.usd")
DRESS_POS    = np.array([0.0, 1.3, 0.8])
DRESS_SCALE  = np.array([0.01, 0.01, 0.01])

SETTLE_STEPS = 250
N_SAMPLES    = args.samples
N_PCD        = 2048
MIN_VISIBLE  = 2048

RAYCAST_DEPTH_TOLERANCE = 0.01   # meters — keep particles within this of the nearest surface
RAYCAST_IMAGE_SIZE      = 512    # virtual depth buffer resolution

# 4 cameras placed ±CAM_XY_OFFSET in X and Y around the garment centroid, all at CAM_Z height.
# Matches folder5000 camera layout for consistency.
CAM_Z        = 1.0   # absolute world height (metres above ground)
CAM_XY_OFFSET = 1.0  # metres from centroid in x or y

OUT_DIR = args.outdir if args.outdir else os.path.join(PROJECT_ROOT, "data/majca/0_majca")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# WORLD SETUP
# ---------------------------------------------------------------------------
world = World(physics_dt=1/120, backend="torch", device="cuda:0")
world.scene.add_ground_plane(size=25.0, color=np.array([0.5, 0.5, 0.5]))
physics = world.get_physics_context()
physics.enable_ccd(True)
physics.enable_gpu_dynamics(True)
physics.set_broadphase_type("gpu")
physics.enable_stablization(True)
stage = world.scene.stage

UsdGeom.Xform.Define(stage, "/World/Garment")
particle_material = ParticleMaterial(prim_path="/World/Garment/particleMaterial", friction=0.8)
particle_system = ParticleSystem(
    prim_path="/World/Garment/particleSystem",
    simulation_owner=world.get_physics_context().prim_path,
    particle_contact_offset=0.008,
    enable_ccd=True,
    global_self_collision_enabled=True,
    non_particle_collision_enabled=True,
    solver_position_iteration_count=32,
)
add_reference_to_stage(usd_path=DRESS_USD, prim_path="/World/Garment/garment")
garment_xform = XFormPrim(
    prim_path="/World/Garment/garment",
    name="garment",
    position=DRESS_POS,
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    scale=DRESS_SCALE,
)
garment_mesh = ClothPrim(
    name="garment_mesh",
    prim_path="/World/Garment/garment/mesh",
    particle_system=particle_system,
    particle_material=particle_material,
    stretch_stiffness=1000.0,
    bend_stiffness=15.0,
    shear_stiffness=5.0,
    spring_damping=2.0,
)

world.reset()
garment_mesh._cloth_prim_view.initialize(world._physics_sim_view)

print("Initial settle...")
for _ in range(300):
    world.step(render=False)

# Get flat reference positions to rotate from
initial_particles = garment_mesh._cloth_prim_view.get_world_positions()
if hasattr(initial_particles, "cpu"):
    initial_particles = initial_particles.cpu().numpy()
if initial_particles.ndim == 3:
    initial_particles = initial_particles.squeeze(0)

print(f"Mesh particle count: {len(initial_particles)}")

# Drop height = half diagonal of bounding box + clearance
bbox = initial_particles.max(axis=0) - initial_particles.min(axis=0)
half_diag = np.linalg.norm(bbox) / 2.0
drop_z = half_diag + 1.0

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def random_quaternion_tilted(min_angle_deg=60):
    tilt    = np.random.uniform(np.radians(min_angle_deg), np.radians(90))
    azimuth = np.random.uniform(0, 2 * np.pi)
    axis    = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    spin    = np.random.uniform(0, 2 * np.pi)
    R1 = Rotation.from_rotvec(axis * tilt)
    R2 = Rotation.from_rotvec(np.array([0.0, 0.0, 1.0]) * spin)
    return (R2 * R1).as_quat()


# Sleeve and hem particles are excluded from the squeeze — they keep their original positions
IGNORE_INDICES = np.array([
    # Left sleeve
    1266, 1267, 1268, 1269, 1287, 1288, 1388, 1438, 1451, 1452, 1453, 1454, 2227, 2301, 2328,
    3587, 3646, 8299, 8300, 8301, 8302, 8320, 8321, 8322, 8502, 8506, 8507, 8508, 8509, 8510,
    8607, 8608, 8609, 8610, 8611, 8618, 8635, 8636, 8637, 8639, 8640, 8641, 8642, 8643, 8644,
    8645, 8646, 8647, 8648, 8649, 8757, 8760, 11152, 11153, 11154, 11270, 11271, 11272, 11273,
    11274, 11275, 11323, 11324, 11325, 11326, 15140, 15141, 15142, 15143, 15144, 15193, 15304,
    15305, 15306, 15307,
    # Right sleeve
    479, 480, 481, 498, 499, 500, 572, 573, 582, 630, 633, 634, 657, 658, 719, 721, 740, 811,
    845, 5990, 5991, 5992, 6010, 6011, 6012, 6013, 6142, 6143, 6144, 6145, 6163, 6164, 6165,
    6166, 6239, 6248, 6249, 6250, 6251, 6252, 6253, 6254, 6255, 6302, 6303, 6305, 6306, 6308,
    6309, 6456, 6457, 6458, 6461, 6463, 6464, 6465, 6466, 6467, 6468, 6527, 6528, 6529, 6531,
    6764, 6765, 6876, 6878, 11171, 11980, 11981, 17217, 17218,
])


def pre_crumple(particles, squeeze):
    c      = particles.mean(axis=0)
    result = c + (particles - c) * squeeze
    result[IGNORE_INDICES] = particles[IGNORE_INDICES]  # sleeves stay at original positions
    return result


def get_particles():
    pts = garment_mesh._cloth_prim_view.get_world_positions()
    if hasattr(pts, "cpu"):
        pts = pts.cpu().numpy()
    if pts.ndim == 3:
        pts = pts.squeeze(0)
    return pts


def reset_garment():
    low_or_high = np.random.choice([0, 1])
    if low_or_high == 0:
        squeeze = np.random.uniform(0.6, 0.95)
    else:
        squeeze = np.random.uniform(1.05, 1.3)
    pts      = pre_crumple(initial_particles, squeeze)
    if np.random.random() < 0.10:
        R = Rotation.from_quat(random_quaternion_tilted(0)).as_matrix()   # flat drop (0–90°)
    else:
        R = Rotation.from_quat(random_quaternion_tilted(60)).as_matrix()  # heavy tilt (60–90°)
    centroid = pts.mean(axis=0)
    rotated  = (pts - centroid) @ R.T + centroid
    rotated[:, 2] += drop_z - rotated[:, 2].min()
    garment_mesh._cloth_prim_view.set_world_positions(
        torch.tensor(rotated, dtype=torch.float32).unsqueeze(0).to("cuda:0")
    )
    for _ in range(SETTLE_STEPS):
        world.step(render=False)


def raycast_occlusion(particles, cam_pos, cam_target, depth_tolerance=0.01, image_size=512):
    """
    Depth-buffer occlusion: rasterize particles onto a virtual image,
    build a per-pixel minimum depth map, keep only particles within
    depth_tolerance of the nearest surface at each pixel.
    Scale-invariant and O(N) — no radius tuning needed.
    """
    rel   = particles - cam_pos
    depth = np.linalg.norm(rel, axis=1)
    dirs  = rel / depth[:, None]

    # Camera basis
    fwd = cam_target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(fwd[2]) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])   # gimbal lock fix for overhead camera
    right = np.cross(fwd, world_up); right /= np.linalg.norm(right)
    up    = np.cross(right, fwd);    up    /= np.linalg.norm(up)

    # Drop particles behind the camera
    in_front  = (dirs @ fwd) > 0.0
    front_idx = np.where(in_front)[0]
    depth_f   = depth[front_idx]
    dirs_f    = dirs[front_idx]

    # Project to virtual image pixels (120° horizontal FOV)
    fov_half = np.radians(60)
    focal    = (image_size / 2.0) / np.tan(fov_half)
    x_cam = dirs_f @ right
    y_cam = dirs_f @ up
    z_cam = dirs_f @ fwd

    u = np.floor(focal * x_cam / z_cam + image_size / 2.0).astype(int)
    v = np.floor(focal * y_cam / z_cam + image_size / 2.0).astype(int)
    in_fov = (u >= 0) & (u < image_size) & (v >= 0) & (v < image_size)

    # Build minimum-depth buffer via vectorised scatter
    depth_buffer = np.full(image_size * image_size, np.inf)
    in_fov_idx   = np.where(in_fov)[0]
    pixel_idx    = v * image_size + u
    np.minimum.at(depth_buffer, pixel_idx[in_fov_idx], depth_f[in_fov_idx])

    # Keep particles within depth_tolerance of the minimum depth at their pixel
    min_depth_at_pixel = depth_buffer[pixel_idx]
    keep = in_fov & (depth_f <= min_depth_at_pixel + depth_tolerance)

    idx = front_idx[np.where(keep)[0]]
    return particles[idx], idx


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
existing  = sorted([f for f in os.listdir(OUT_DIR) if f.endswith(".npz")])
collected = int(existing[-1].split("_")[1].split(".")[0]) + 1 if existing else 0
attempts  = 0
target    = collected + N_SAMPLES
if existing:
    print(f"Resuming from sample {collected} ({len(existing)} already saved)")

print(f"\nGenerating {N_SAMPLES} samples → {OUT_DIR}\n")

while collected < target:
    attempts += 1

    reset_garment()
    mesh_pts = get_particles()
    centroid = mesh_pts.mean(axis=0)

    cam_positions = [
        np.array([centroid[0] + CAM_XY_OFFSET, centroid[1],               CAM_Z]),
        np.array([centroid[0] - CAM_XY_OFFSET, centroid[1],               CAM_Z]),
        np.array([centroid[0],                  centroid[1] + CAM_XY_OFFSET, CAM_Z]),
        np.array([centroid[0],                  centroid[1] - CAM_XY_OFFSET, CAM_Z]),
    ]

    for cam_pos in cam_positions:
        if collected >= target:
            break

        visible_pts, visible_idx = raycast_occlusion(
            mesh_pts, cam_pos, centroid,
            depth_tolerance=RAYCAST_DEPTH_TOLERANCE,
            image_size=RAYCAST_IMAGE_SIZE,
        )

        pct = 100 * len(visible_pts) / len(mesh_pts)
        print(f"[{collected+1}/{N_SAMPLES}] attempt {attempts} cam=({cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f}) | "
              f"visible={len(visible_pts)} ({pct:.0f}%) | "
              f"centroid={np.round(centroid, 3)}", end="")

        if len(visible_pts) < MIN_VISIBLE:
            print(f"  → skipped (< {MIN_VISIBLE} visible)")
            continue

        # FPS downsample visible points to N_PCD
        pcd_pts = furthest_point_sampling(visible_pts, n_samples=N_PCD)
        if hasattr(pcd_pts, "cpu"):
            pcd_pts = pcd_pts.cpu().numpy()

        # Normalize to centroid — model learns shape, not world position
        pcd_centroid = pcd_pts.mean(axis=0)
        pcd_pts_normalized = pcd_pts - pcd_centroid

        # Save .npz
        out_path = os.path.join(OUT_DIR, f"majca_{collected:04d}.npz")
        np.savez(
            out_path,
            mesh_points          = mesh_pts.astype(np.float32),
            pcd_points           = pcd_pts_normalized.astype(np.float32),
            pcd_centroid         = pcd_centroid.astype(np.float32),
            visible_mesh_indices = visible_idx.astype(np.int32),
        )

        print(f"  → saved {os.path.basename(out_path)}")
        collected += 1

print(f"\nDone. {collected} samples in {attempts} attempts.")
print(f"Saved to: {OUT_DIR}")
simulation_app.close()
