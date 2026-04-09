import os
import sys

import numpy as np
import torch
import open3d as o3d
import random
from unigarmentmanip.model.pointnet2_UniGarmentManip import UniGarmentManip_Model
from Env.Utils.pointcloud import furthest_point_sampling, normalize_pcd_points

class UniGarmentManip_Encapsulation:

    def __init__(self, catogory:str="Tops_LongSleeve"):
        '''
        load model
        '''
        self.catogory = catogory
        # set resume path
        checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
        if catogory == "majca":
            resume_path = os.path.join(checkpoints_dir, "majca", "checkpoint_epoch_8.pth")
        else:
            resume_path = os.path.join(checkpoints_dir, self.catogory, "checkpoint.pth")
        # set seed
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # define model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = UniGarmentManip_Model(normal_channel=False, feature_dim=512).cuda()
        ckpt = torch.load(resume_path, weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()


    def get_feature(self, input_pcd:np.ndarray, index_list:list=None):
        '''
        get feature of input point cloud
        '''
        normalized_pcd, *_ = normalize_pcd_points(input_pcd)
        normalize_pcd = np.expand_dims(normalized_pcd, axis=0)

        with torch.no_grad():

            pcd_features = self.model(
                torch.from_numpy(normalize_pcd).to(self.device).float(),
            ).squeeze(0)
            # print(pcd_features.shape)

        if index_list is not None:
            target_features_list = []
            for i in index_list:
                target_features_list.append(pcd_features[i])
            return torch.stack(target_features_list)
        else:
            return pcd_features

    def get_manipulation_points(self, input_pcd:np.ndarray, index_list:list=None):
        '''
        get manipulation points of input point cloud.
        index_list items can be a single int or a list of ints (group).
        If a group is given, all group members are matched and their
        results are averaged into one grasp point.
        '''

        checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
        custom_ref  = os.path.join(checkpoints_dir, self.catogory, f"{self.catogory}_flat_reference.ply")
        default_ref = os.path.join(checkpoints_dir, self.catogory, "demo_garment.ply")
        ref_ply     = custom_ref if os.path.exists(custom_ref) else default_ref

        demo_pcd   = np.asarray(o3d.io.read_point_cloud(ref_ply).points)
        input_feat = self.get_feature(input_pcd)
        input_feat = torch.nn.functional.normalize(input_feat, p=2, dim=1)

        grasp_points  = []
        grasp_indices = []

        for item in index_list:
            group = [item] if isinstance(item, int) else item

            group_feat = self.get_feature(demo_pcd, group)
            group_feat = torch.nn.functional.normalize(group_feat, p=2, dim=1)

            scores      = torch.matmul(group_feat, input_feat.T)
            best_values, best_idx = torch.max(scores, dim=1)
            best_idx    = best_idx.detach().cpu().numpy()
            best_values = best_values.detach().cpu().numpy()

            # Pick the single group member with the highest confidence score
            top_member  = np.argmax(best_values)
            grasp_point = input_pcd[best_idx[top_member]]

            print(f"Group {group[:1]}... | best_score={best_values[top_member]:.3f} | grasp={np.round(grasp_point, 4)}")

            grasp_points.append(grasp_point)
            grasp_indices.append(best_idx[0])

        return np.array(grasp_points), np.array(grasp_indices)
