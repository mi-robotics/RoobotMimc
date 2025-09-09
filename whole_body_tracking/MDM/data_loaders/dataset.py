import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from typing import List, Dict
import joblib

OBS_TERM_MAPPING = {
    'actions': (0, 29),
    'base_lin_vel': (29, 32),
    'base_ang_vel': (32, 35),
    'joint_pos': (35, 64),
    'joint_vel': (64, 93),
    'gravity': (93, 96),
    'root_h': (96, 97),
    'local_body_pos': (97, 139),
    'local_body_rots': (139, 223),
    'local_body_vel': (223, 265),
    'local_body_ang_vel': (265, 307),
    'root_xy_position': (307, 309),
    'root_yaw': (309, 310)
}

VAR_TYPE_MAPPING = {
    'actions': False,
    'base_lin_vel': True,
    'base_ang_vel': True,
    'joint_pos': False,
    'joint_vel': False,
    'gravity': True,
    'root_h': True,
    'local_body_pos': True,
    'local_body_rots': True,
    'local_body_vel': True,
    'local_body_ang_vel': True,
    'root_xy_position': True,
    'root_yaw': True
}

def get_features_dims(keys):
    total = 0
    for k in keys:
        total += OBS_TERM_MAPPING[k][1]-OBS_TERM_MAPPING[k][0]

    return total

def extract_observation_terms(
    data: np.ndarray|torch.Tensor, keys: List[str]
) -> np.ndarray:

    
    if data.ndim != 2 or data.shape[1] != 310:
        raise ValueError(
            f"Expected data shape (num_frames, 307), but got {data.shape}"
        )

    # A list to hold the slices of data for concatenation
    data_slices = []
    
    for key in keys:
        if key in OBS_TERM_MAPPING:
            start_idx, end_idx = OBS_TERM_MAPPING[key]
            d = data[:, start_idx:end_idx]

            if key == 'root_xy_position':
                d_0 = d[[0], :] #seq len , features 

                if isinstance(d, np.ndarray):
                    d = np.concatenate((d_0, d))
                elif isinstance(d, torch.Tensor):
                    d = torch.cat((d_0, d))
            
                d = d[1:] - d[:-1] # seq len ,  feature

            data_slices.append(d)

        else:
            ValueError(f"Warning: Key '{key}' not found. It will be skipped.")

    if isinstance(data, np.ndarray):
        return np.concatenate(data_slices, axis=1)
    elif isinstance(data, torch.Tensor):
        return torch.cat(data_slices, dim=1)
        



# This is the main function to solve your problem.
def calculate_statistics_for_subset(
    subset_dataset: List[np.ndarray], keys: List[str]
) -> Dict[str, Dict[str, np.ndarray]]:
    
    if not subset_dataset:
        raise ValueError("Input dataset list cannot be empty.")

    # Step 2: Concatenate the list of subset arrays into one large array.
    full_subset_data = np.concatenate(subset_dataset, axis=0)
    print(f"Shape of the full concatenated subset data: {full_subset_data.shape}")

    means = np.zeros((full_subset_data.shape[1],))    
    stds = np.zeros((full_subset_data.shape[1],))

    current_index = 0
    for key in keys:
        if key not in OBS_TERM_MAPPING:
            continue
            
        # Determine the number of columns for this feature
        original_start, original_end = OBS_TERM_MAPPING[key]
        feature_width = original_end - original_start
        
        # Define the slice in the new subset array
        start_idx = current_index
        end_idx = current_index + feature_width
        
        # Extract the data for this feature from the concatenated subset
        feature_data = full_subset_data[:, start_idx:end_idx]
        
        # Calculate and store the statistics
        mean = np.mean(feature_data, axis=0)
        std = np.std(feature_data, axis=0)

        if key == 'root_yaw':
            print(mean, std, np.min(feature_data, axis=0), np.max(feature_data, axis=0))

        if VAR_TYPE_MAPPING[key]:
            std = std.mean()
   
        means[start_idx:end_idx] = mean
        stds[start_idx:end_idx] = std

        # Move the index for the next feature
        current_index = end_idx


    return means, stds



'''For use of training text-2-motion generative model'''
class RobotStateActionDataset(data.Dataset):
    def __init__(self, cfg):
        '''
        +--------------------------------------------------------------------+
        | Active Observation Terms in Group: 'data_collection_obs' (shape: (307,)) |
        +-------------+----------------------------------------+-------------+
        |    Index    | Name                                   |    Shape    |
        +-------------+----------------------------------------+-------------+
        |      0      | actions                                |    (29,)    |
        |      1      | base_lin_vel                           |     (3,)    |
        |      2      | base_ang_vel                           |     (3,)    |
        |      3      | joint_pos                              |    (29,)    |
        |      4      | joint_vel                              |    (29,)    |
        |      5      | gravity                                |     (3,)    |
        |      6      | root_h                                 |     (1,)    |
        |      7      | local_body_pos                         |    (42,)    |
        |      8      | local_body_rots                        |    (84,)    |
        |      9      | local_body_vel                         |    (42,)    |
        |      10     | local_body_ang_vel                     |    (42,)    |
        |      11     | root_xy_position                       |     (2,)    |
        |      12     | root_yaw                               |     (1,)    |
        +-------------+----------------------------------------+-------------+
        '''
        self.cfg = cfg

        self.pred_len = cfg.model.pred_len
        self.context_len = cfg.model.context_len
        self.corrupt_context = cfg.dataset.corrupt_context

        self.max_length = self.pred_len+self.context_len

        self.velocity_indexes = [0,1,2]

        if self.cfg.dataset.load_data:
            self.load_data()
        
        if self.cfg.dataset.load_mean_std:
            self.load_mean_std()



    def load_data(self):
        print('LOADING DATATSET ========')

        data = joblib.load(self.cfg.dataset.data_file)  

        state_data_keys = self.cfg.dataset.state_data_keys
        context_data_keys = self.cfg.dataset.context_data_keys
        action_data_keys = self.cfg.dataset.action_data_keys

        
        self.state_data = [extract_observation_terms(d['motion'], keys=state_data_keys) for d in data]
        self.s_mean, self.s_std = calculate_statistics_for_subset(self.state_data, keys=state_data_keys)

        self.context_data = [extract_observation_terms(d['motion'], keys=context_data_keys) for d in data]
        self.c_mean, self.c_std = calculate_statistics_for_subset(self.context_data, keys=context_data_keys)

        self.action_data = [extract_observation_terms(d['motion'], keys=action_data_keys) for d in data]
        self.a_mean, self.a_std = calculate_statistics_for_subset(self.action_data, keys=action_data_keys)

        self.captions = [d['caption'] for d in data]       
        #TODO fix this in the data collection  
        self.lengths = [d['length'].item() for d in data]


        data_out = {
            'a_mean':self.a_mean, 'a_std':self.a_std, 
            's_mean': self.s_mean, 's_std': self.s_std,
            'c_mean': self.c_mean, 'c_std':self.c_std
        }
        data_name = self.cfg.dataset.data_file.split('/')[-1].replace('.pkl', '_mean_std.pkl')
        joblib.dump(data_out, self.cfg.training.save_dir + '/' + data_name)

        print('State shapes:=====', self.state_data[0].shape, self.s_mean.shape, self.s_std.shape)       
        print('Context shapes:=====', self.context_data[0].shape, self.c_mean.shape, self.c_std.shape)
        print('Action shapes:=====', self.action_data[0].shape, self.a_mean.shape, self.a_std.shape)

        print('LOADED DATATSET ========')

        del data

    def load_mean_std(self):
        data_name = self.cfg.dataset.data_file.split('/')[-1].replace('.pkl', '_mean_std.pkl')
        data = joblib.load( 
            '/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/MDM/log/robot_obs_bm_target_vel_2025-09-02_17-01-17/robot_sa_dataset_r5_mean_std.pkl'
            # self.cfg.training.save_dir + '/' + data_name
            )
        self.a_mean, self.a_std = data['a_mean'], data['a_std']        
        self.c_mean, self.c_std = data['c_mean'], data['c_std']
        self.s_mean, self.s_std = data['s_mean'], data['s_std']



    def empty_data(self):
        self.state_data = None        
        self.action_data = None
        self.context_data = None
        self.lengths = None
        self.captions = None
        return


    def inv_transform(self, data):
        raise NotImplementedError
        return data * self.std + self.mean
    
    def inv_transform_s(self, data):
        if isinstance(data, torch.Tensor):
            return data \
                * torch.from_numpy(self.s_std).to(data.device)[None, None, :, None] \
                    + torch.from_numpy(self.s_mean).to(data.device)[None, None, :, None]
    
        return data * self.s_std + self.s_mean

    
    def inv_transform_a(self, data):
        if isinstance(data, torch.Tensor):
            return data \
                * torch.from_numpy(self.a_std).to(data.device)[None, None, :, None] \
                    + torch.from_numpy(self.a_mean).to(data.device)[None, None, :, None]
        return data * self.a_std + self.a_mean
    

    def compute_velocity_target(self, data):
        total_delta = np.sum(data, axis=0)
        target = total_delta/len(data)*50 
        return target


    def __len__(self):
        return len(self.action_data) 

    def __getitem__(self, idx):
        # Randomly select a caption
        text_list = self.captions[idx]
        m_length = self.lengths[idx]

        #TODO need to fix the 
        caption = 'some text'#random.choice(text_list)

        motion_frame_idx = random.randint(0, m_length -1 - self.max_length)
        a = self.action_data[idx][motion_frame_idx : motion_frame_idx + self.max_length]        
        c = self.context_data[idx][motion_frame_idx : motion_frame_idx + self.max_length]
        s = self.state_data[idx][motion_frame_idx : motion_frame_idx + self.max_length]
        
        target_vels = self.compute_velocity_target(s[:, self.velocity_indexes])
        "Z Normalization"
        a = (a - self.a_mean) / self.a_std        
        c = (c - self.c_mean) / self.c_std
        s = (s - self.s_mean) / self.s_std


        return a, c, s, target_vels, m_length, caption

