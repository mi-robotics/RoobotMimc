


cd whole_body_tracking



python scripts/rsl_rl/train.py --task=Large-Teacher-G1-Multi-Futures-v0 \
    --registry_name null \
    --headless --logger wandb --log_project_name tracking --run_name multi_amass_simple \
    --motion_file '/home/mcarroll/Documents/cd-2/humanoid_tracking/data/amass_simple/packaged.pkl' \
    --max_iterations 100000 \
    agent.save_interval=5000