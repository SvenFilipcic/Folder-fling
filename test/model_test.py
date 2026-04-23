import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import open3d as o3d
from unigarmentmanip.model.UniGarmentManip_Encapsulation import UniGarmentManip_Encapsulation

# Sleeve groups on the majca hires flat reference (index + 20 nearest neighbors)
LEFT_GROUP  = [1266, 1267, 1268, 1269, 1287, 1288, 1388, 1438, 1451, 1452, 1453, 1454, 2227, 2301, 2328, 3587, 3646, 8299, 8300, 8301, 8302, 8320, 8321, 8322, 8502, 8506, 8507, 8508, 8509, 8510, 8607, 8608, 8609, 8610, 8611, 8618, 8635, 8636, 8637, 8639, 8640, 8641, 8642, 8643, 8644, 8645, 8646, 8647, 8648, 8649, 8757, 8760, 11152, 11153, 11154, 11270, 11271, 11272, 11273, 11274, 11275, 11323, 11324, 11325, 11326, 15140, 15141, 15142, 15143, 15144, 15193, 15304, 15305, 15306, 15307]
RIGHT_GROUP = [479, 480, 481, 498, 499, 500, 572, 573, 582, 630, 633, 634, 657, 658, 719, 721, 740, 811, 845, 5990, 5991, 5992, 6010, 6011, 6012, 6013, 6142, 6143, 6144, 6145, 6163, 6164, 6165, 6166, 6239, 6248, 6249, 6250, 6251, 6252, 6253, 6254, 6255, 6302, 6303, 6305, 6306, 6308, 6309, 6456, 6457, 6458, 6461, 6463, 6464, 6465, 6466, 6467, 6468, 6527, 6528, 6529, 6531, 6764, 6765, 6876, 6878, 11171, 11980, 11981, 17217, 17218]


def flat_to_standing(points):
    out = points.copy()
    out[:, 1] = points[:, 2]
    out[:, 2] = points[:, 1]
    return out


def make_sphere(center, radius=0.018, color=(1, 0, 0)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(list(color))
    sphere.compute_vertex_normals()
    return sphere


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start",      type=int, default=0,    help="Start from this npz index")
parser.add_argument("--count",      type=int, default=10,   help="Number of samples to show")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth (default: latest majca)")
args = parser.parse_args()

NPZ_FOLDER = os.path.join(PROJECT_ROOT, "data/majca/majca_1")

all_files = sorted([f for f in os.listdir(NPZ_FOLDER) if f.endswith(".npz")])
files_to_run = all_files[args.start:args.start + args.count]

print("Loading model...")
manip = UniGarmentManip_Encapsulation(catogory="majca", checkpoint_override=args.checkpoint)

for file_name in files_to_run:
    npz_path = os.path.join(NPZ_FOLDER, file_name)
    data     = np.load(npz_path)
    points   = data["pcd_points"]
    xyz      = points[:, :3]
    print(f"\n--- {file_name} ({len(points)} pts) ---")

    _, indices = manip.get_manipulation_points(
        input_pcd=points,
        index_list=[LEFT_GROUP, RIGHT_GROUP],
    )

    left_grasp  = xyz[indices[0]]
    right_grasp = xyz[indices[1]]
    print(f"Left grasp:  {np.round(left_grasp,  4)}")
    print(f"Right grasp: {np.round(right_grasp, 4)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([0.2, 0.6, 1.0])

    o3d.visualization.draw_geometries(
        [pcd,
         make_sphere(left_grasp,  color=(0.0, 0.9, 0.1)),
         make_sphere(right_grasp, color=(1.0, 0.1, 0.1))],
        window_name=file_name,
        width=900,
        height=700,
    )
