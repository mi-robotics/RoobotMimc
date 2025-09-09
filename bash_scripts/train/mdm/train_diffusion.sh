cd whole_body_tracking/MDM

export PYTHONPATH="/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/MDM:$PYTHONPATH"

# python ./train/train_mdm.py --config-name=beyond_mimic run_name='bm_state_0' dataset.data_file='/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-26_10-49-04_lefan_walk/collection/v2/0.0/robot_sa_dataset_r5.pkl'

# python ./train/train_mdm.py --config-name=beyond_mimic run_name='bm_state_0.025' dataset.data_file='/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-26_10-49-04_lefan_walk/collection/v2/0.025/robot_sa_dataset_r5.pkl'

python ./train/train_mdm.py \
    --config-name=beyond_mimic_robot_obs_vel_targets run_name='robot_obs_bm_target_vel' \
    model.attn_type='causal' \
    dataset.data_file='/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-26_10-49-04_lefan_walk/collection/v3/0.1/robot_sa_dataset_r5.pkl'

# python ./train/train_mdm.py \
#     --config-name=beyond_mimic run_name='bm_state_0.05_full' \
#     model.attn_type='full' \
#     dataset.data_file='/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-26_10-49-04_lefan_walk/collection/v2/0.05/robot_sa_dataset_r5.pkl'


# python ./train/train_mdm.py --config-name=beyond_mimic run_name='bm_state_0.1_no_rots' dataset.data_file='/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-26_10-49-04_lefan_walk/collection/v2/0.1/robot_sa_dataset_r5.pkl'