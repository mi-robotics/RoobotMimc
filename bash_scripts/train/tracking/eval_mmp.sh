


cd whole_body_tracking


export CUDA_LAUNCH_BLOCKING=1
python scripts/rsl_rl/play.py --task=Large-Teacher-G1-Multi-Futures-v0 \
    --num_envs 10 \
    --evaluate_metrics \
    # --headless
    # --motion_file '/home/mcarroll/Documents/cd-2/humanoid_tracking/data/amass_simple/packaged.pkl' 