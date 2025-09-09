cd whole_body_tracking/consistency_models

export PYTHONPATH="/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/MDM:$PYTHONPATH"
export PYTHONPATH="/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/MDconsistency_modelsM:$PYTHONPATH"


python ./scripts/edm_train.py \
    --config-name=base_edm run_name='target_vel_weighted' \
    model.attn_type='causal' \
    dataset.data_file='/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-26_10-49-04_lefan_walk/collection/v3/0.05/robot_sa_dataset_r5.pkl'
