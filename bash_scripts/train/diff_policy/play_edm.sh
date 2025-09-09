

cd whole_body_tracking

export PYTHONPATH="/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/MDM:$PYTHONPATH"
export PYTHONPATH="/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/MDconsistency_modelsM:$PYTHONPATH"

python scripts/rsl_rl/play_diffusion.py \
    --task=Teacher-G1-Multi-v0 \
    --num_envs 1 \
    --diffussion_type 'edm' \
    --diffusion_loadrun 'robot_obs_bm_target_vel_2025-09-02_17-01-17' \
    --diffusion_cp 600000