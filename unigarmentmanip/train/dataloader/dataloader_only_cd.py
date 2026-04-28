import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader.utils import get_deformation_paths, create_cross_deformation_pairs, create_cross_object_pairs, fps_with_selected, nearest_mesh2pcd, create_cross_only_deformation_pairs
from base.config import Config

configs = Config()
configs = configs.data_config

class Dataset(Dataset):
    
    def __init__(self, mode):
        
        assert mode in ['train', 'val']
        
        self.all_deform_paths = get_deformation_paths(configs.only_deformation_data_dir, configs.garment_data_num, configs.train_ratio, mode)
        self.cross_deformation_pair_path = create_cross_only_deformation_pairs(self.all_deform_paths)
        
        print(self.all_deform_paths[0][0])
        
        print(f"Number of all deformation paths: {len(self.all_deform_paths)}")
        print(f"Number of cross deformation pairs: {len(self.cross_deformation_pair_path)}")
        
    def __len__(self):
        return len(self.cross_deformation_pair_path)
    
    def __getitem__(self, index):
        
        return self.get_cross_deformation_pair(index)
    
    def get_cross_deformation_correspondence(self, index):
        npz1, npz2 = self.cross_deformation_pair_path[index]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)

        # Primary grasp groups — sleeve tips (index + neighbors on majca hires flat reference)
        LEFT_SLEEVE_GROUP  = [1266, 1267, 1268, 1269, 1287, 1288, 1388, 1438, 1451, 1452, 1453, 1454, 2227, 2301, 2328, 3587, 3646, 8299, 8300, 8301, 8302, 8320, 8321, 8322, 8502, 8506, 8507, 8508, 8509, 8510, 8607, 8608, 8609, 8610, 8611, 8618, 8635, 8636, 8637, 8639, 8640, 8641, 8642, 8643, 8644, 8645, 8646, 8647, 8648, 8649, 8757, 8760, 11152, 11153, 11154, 11270, 11271, 11272, 11273, 11274, 11275, 11323, 11324, 11325, 11326, 15140, 15141, 15142, 15143, 15144, 15193, 15304, 15305, 15306, 15307]
        RIGHT_SLEEVE_GROUP = [479, 480, 481, 498, 499, 500, 572, 573, 582, 630, 633, 634, 657, 658, 719, 721, 740, 811, 845, 5990, 5991, 5992, 6010, 6011, 6012, 6013, 6142, 6143, 6144, 6145, 6163, 6164, 6165, 6166, 6239, 6248, 6249, 6250, 6251, 6252, 6253, 6254, 6255, 6302, 6303, 6305, 6306, 6308, 6309, 6456, 6457, 6458, 6461, 6463, 6464, 6465, 6466, 6467, 6468, 6527, 6528, 6529, 6531, 6764, 6765, 6876, 6878, 11171, 11980, 11981, 17217, 17218]

        # Fallback grasp groups — bottom hem corners (used when sleeves not visible)
        LEFT_HEM_GROUP  = [239, 240, 335, 336, 357, 377, 378, 397, 5384, 5648, 5649, 5650, 5651, 5673, 5674, 5675, 5723, 5724, 5726, 5727, 5728, 5729, 5730, 5794, 5795, 5796]
        RIGHT_HEM_GROUP = [21, 22, 127, 128, 143, 155, 4793, 4794, 4823, 4824, 5075, 5076, 5092, 5093, 5094, 5095, 5096, 5118, 5119, 5120, 5121, 5122, 5123, 5170, 5172, 5251]

        SLEEVE_INDICES = LEFT_SLEEVE_GROUP + RIGHT_SLEEVE_GROUP
        HEM_INDICES    = LEFT_HEM_GROUP + RIGHT_HEM_GROUP
        OVERSAMPLE     = 3  # repeat priority correspondences this many times

        random_correspondence_num = configs.correspondence_num
        mesh_points_1 = npz1['mesh_points']
        mesh_points_2 = npz2['mesh_points']
        pcd_points_1 = npz1['pcd_points']
        pcd_points_2 = npz2['pcd_points']
        visible_mesh_id_1 = npz1['visible_mesh_indices']
        visible_mesh_id_2 = npz2['visible_mesh_indices']
        visible_mesh_id = np.intersect1d(visible_mesh_id_1, visible_mesh_id_2)

        sleeve_visible = [i for i in SLEEVE_INDICES if i in visible_mesh_id]
        hem_visible    = [i for i in HEM_INDICES    if i in visible_mesh_id]

        # Use sleeves as priority seed; fall back to hem if no sleeves visible in both clouds
        priority_visible = sleeve_visible if sleeve_visible else hem_visible

        random_points_id, random_points = fps_with_selected(mesh_points_1, visible_mesh_id, priority_visible, random_correspondence_num)
        pcd_random_points_id_1 = nearest_mesh2pcd(mesh_points_1, pcd_points_1, random_points_id)
        pcd_random_points_id_2 = nearest_mesh2pcd(mesh_points_2, pcd_points_2, random_points_id)

        keypoints_id_visible_1 = np.array(pcd_random_points_id_1)
        keypoints_id_visible_2 = np.array(pcd_random_points_id_2)

        correspondence = np.stack([keypoints_id_visible_1, keypoints_id_visible_2], axis=1)

        # Oversample the priority group so the model gets extra gradient on those points
        if priority_visible:
            priority_pcd_id_1 = nearest_mesh2pcd(mesh_points_1, pcd_points_1, priority_visible)
            priority_pcd_id_2 = nearest_mesh2pcd(mesh_points_2, pcd_points_2, priority_visible)
            priority_corr = np.stack([np.array(priority_pcd_id_1), np.array(priority_pcd_id_2)], axis=1)
            priority_corr_repeated = np.tile(priority_corr, (OVERSAMPLE, 1))
            correspondence = np.vstack([correspondence, priority_corr_repeated])

        # Shuffle and truncate to fixed size so all samples collate into a batch
        np.random.shuffle(correspondence)
        correspondence = correspondence[:random_correspondence_num]

        return correspondence
    
    
    def get_cross_deformation_pair(self, index):

        npz1, npz2 = self.cross_deformation_pair_path[index]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)
        pc1 = npz1['pcd_points']
        pc2 = npz2['pcd_points']
        correspondence = self.get_cross_deformation_correspondence(index)

        return pc1, pc2, correspondence



if __name__ == '__main__':
    
    dataset = Dataset('train')
    print(dataset.__len__())
    pc1, pc2, correspondence= dataset[1]
    print(pc1.shape)
    print(pc2.shape)
    print(correspondence.shape)
    # print(correspondence)
    
    from utils import visualize_point_cloud
    visualize_point_cloud(pc1, correspondence[:, 0], title="Point Cloud 1")
    visualize_point_cloud(pc2, correspondence[:, 1], title="Point Cloud 2")
    
    
    for i in range(configs.correspondence_num):
        visualize_point_cloud(pc1, correspondence[i:i+1, 0], title="Point Cloud 1")
        visualize_point_cloud(pc2, correspondence[i:i+1, 1], title="Point Cloud 2")
    