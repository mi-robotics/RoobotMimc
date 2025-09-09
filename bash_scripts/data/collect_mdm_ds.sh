
cd whole_body_tracking

export CUDA_LAUNCH_BLOCKING=1
# python scripts/rsl_rl/play.py --task=Teacher-G1-Multi-v0 \
#     --load_run '2025-08-26_10-49-04_lefan_walk' \
#     --num_envs 4096 \
#     --collect_dataset \
#     --action_noise_level 0.1 \
#     --headless


python scripts/rsl_rl/play.py --task=Teacher-G1-Multi-v0 \
    --load_run '2025-08-26_10-49-04_lefan_walk' \
    --num_envs 4096 \
    --collect_dataset \
    --action_noise_level 0.1 \
    --headless


# python scripts/rsl_rl/play.py --task=Teacher-G1-Multi-v0 \
#     --load_run '2025-08-26_10-49-04_lefan_walk' \
#     --num_envs 4096 \
#     --collect_dataset \
#     --action_noise_level 0.025 \
#     --headless


# python scripts/rsl_rl/play.py --task=Teacher-G1-Multi-v0 \
#     --load_run '2025-08-26_10-49-04_lefan_walk' \
#     --num_envs 4096 \
#     --collect_dataset \
#     --action_noise_level 0.0 \
#     --headless