

cd whole_body_tracking


# python scripts/package_data.py \
#     --motion_dir /home/mcarroll/Documents/cd-2/humanoid_tracking/data/simple_walk/poses \
#     --input_fps 30 \
#     --output_path /home/mcarroll/Documents/cd-2/humanoid_tracking/data/simple_walk/packaged.pkl \
#     --headless


python scripts/package_data.py \
    --motion_dir /home/mcarroll/Documents/cd-2/humanoid_tracking/data/amass_simple/poses \
    --input_fps 30 \
    --output_path /home/mcarroll/Documents/cd-2/humanoid_tracking/data/amass_simple/packaged.pkl \
    --headless