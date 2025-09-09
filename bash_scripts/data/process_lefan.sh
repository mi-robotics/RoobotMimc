
cd whole_body_tracking

# python scripts/csv_to_npz.py --input_file /home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv \
#     --input_fps 30 --output_name padded_walk1_subject1 --headless --pad_motions

python scripts/csv_to_npz.py --input_file /home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv \
    --input_fps 30 --output_name walk1_subject1 --headless