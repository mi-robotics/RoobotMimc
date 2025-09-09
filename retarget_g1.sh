
which pip

# which pip

cd ProtoMotions

# pip install -e .
# pip install -e isaac_utils
# pip install -e poselib
# pip install -r requirements_isaacgym.txt
# pip install ipdb
# pip install mink
# pip install dm_control
# pip install numpy==1.23.5
# pip install loop_rate_limiters
# pip install "qpsolvers[quadprog]"
# pip install open3d
# pip install mujoco
# pip install --upgrade --force-reinstall mujoco

# pip install numpy==1.23.5
# pip install chumpy
pip install pandas

# python ./data/scripts/convert_amass_to_isaac.py \
#     ../data/AMASS/amass_unzipped \
#     --humanoid-type smpl \
#     --robot-type g1 \
#     --force-remake \
#     --force-retarget \
#     --generate-flipped 


python ./data/scripts/retargeting/mink_retarget.py \
    /home/mcarroll/Documents/AMASS/KIT/3/walk_6m_straight_line04_poses.npz \
    ./0-KIT_3_walk_6m_straight_line04_poses.csv \
    g1 \
    --render



# mjpython data/scripts/retargeting/mink_retarget.py   "/Users/bytedance/Documents/humanoid_projects/human_imitation/data/AMASS/amass_unzipped/ACCAD/Female1Gestures_c3d/D5 - Random Stuff 2_poses.npz"   ./test   g1_aneal   --renderwhich