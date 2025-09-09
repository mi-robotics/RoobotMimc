from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch
import pandas as pd

import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.log_data = []  # 存储每一步的日志

        self.config = config
        self.remote_controller = RemoteController()
       
        self.leg_length = 0.17734 + 0.30001  # thigh + shank

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        # self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        # self.target_dof_pos = config.default_angles.copy()
        self.target_dof_pos = np.concatenate([
            config.default_angles,                            # 0–11: Leg joints (12)
            config.arm_waist_target[[0, 1, 2]],               # 12–14: Waist yaw, roll, pitch
            config.arm_waist_target[[3, 4, 6]],               # 15–17: Left shoulder pitch, roll, elbow
            config.arm_waist_target[[10, 11, 13]]             # 18–20: Right shoulder pitch, roll, elbow
        ]).copy()
        # 构造 PD 参数与 controlled_joint2motor_idx 对齐
        self.controlled_kps = (
            self.config.kps + 
            [self.config.arm_waist_kps[i] for i in [0, 1, 2, 3, 4, 6, 10, 11, 13]]
        )
        self.controlled_kds = (
            self.config.kds + 
            [self.config.arm_waist_kds[i] for i in [0, 1, 2, 3, 4, 6, 10, 11, 13]]
        )
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # self.obs = np.zeros(53, dtype=np.float32)
        # self.obs = np.zeros(65, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0


        self.joint_name_by_motor = {
            0: "left_hip_roll",
            1: "left_hip_pitch",
            2: "left_knee",
            3: "left_ankle_pitch",
            4: "left_ankle_roll",
            5: "left_hip_yaw",
            6: "right_hip_roll",
            7: "right_hip_pitch",
            8: "right_knee",
            9: "right_ankle_pitch",
            10: "right_ankle_roll",
            11: "right_hip_yaw",
            12: "waist_yaw",
            13: "waist_roll",
            14: "waist_pitch",
            15: "left_shoulder_pitch",
            16: "left_shoulder_roll",
            17: "left_shoulder_yaw",
            18: "left_elbow",
            19: "left_wrist_yaw",
            20: "left_wrist_roll",
            21: "left_wrist_pitch",
            22: "right_shoulder_pitch",
            23: "right_shoulder_roll",
            24: "right_shoulder_yaw",
            25: "right_elbow",
            26: "right_wrist_yaw",
            27: "right_wrist_roll",
            28: "right_wrist_pitch",
            # ... fill up to 34 if needed
        }

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def update_command(self):
        """
        Scheduled command velocity profile with 15s initial standstill,
        followed by a velocity sequence, then returning to 0 velocity.
        """
        t = self.counter * self.config.control_dt  # 当前时间（秒）
        dt = self.config.control_dt
        step = self.counter

        # --- Define trajectory ---
        vel_seq = [0.0, 0.5, 1.0, 0.7, 0.5, 0.0]
        vel_seq_yaw = [0.0, 0.1, 0.1, 0.05, 0.0]

        # --- Time offset for initial stand ---
        initial_stand_duration = 15.0  # seconds
        t_adjusted = max(t - initial_stand_duration, 0.0)

        # --- Index into scheduled phase ---
        step_per_phase = int(1.0 / dt)
        idx = min(int(t_adjusted // 1.0), len(vel_seq) - 1)
        idx_yaw = min(int(t_adjusted // 1.0), len(vel_seq_yaw) - 1)

        # --- Select velocities ---
        vx = vel_seq[idx]
        yaw_rate = vel_seq_yaw[idx_yaw]

        # If still in initial stand phase or past end of trajectory → hold at 0
        if t < initial_stand_duration or idx >= len(vel_seq) - 1:
            vx = 0.0
            yaw_rate = 0.0

        # --- Assign to command buffer ---
        self.cmd[0] = vx
        self.cmd[1] = 0.0
        self.cmd[2] = yaw_rate

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity

        self.controlled_joint2motor_idx = (
            self.config.leg_joint2motor_idx +
            [self.config.arm_waist_joint2motor_idx[i] for i in [0, 1, 2, 3, 4, 6, 10, 11, 13]]
        )

        for i, motor_idx in enumerate(self.controlled_joint2motor_idx):
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq
        # for i in range(len(self.config.leg_joint2motor_idx)):
        #     self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
        #     self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # self.qj[12] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
        # self.dqj[12] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
        # self.qj[13] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[1]].q
        # self.dqj[13] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[1]].dq
        # self.qj[14] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[2]].q
        # self.dqj[14] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[2]].dq
        # self.qj[15] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[3]].q
        # self.dqj[15] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[3]].dq
        # self.qj[16] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[4]].q
        # self.dqj[16] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[4]].dq

        # self.qj[17] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[6]].q
        # self.dqj[17] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[6]].dq
        # self.qj[18] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[10]].q
        # self.dqj[18] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[10]].dq
        # self.qj[19] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[11]].q
        # self.dqj[19] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[11]].dq
        # self.qj[20] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[12]].q
        # self.dqj[20] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[12]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # ---- 先生成基本21dof (腿12+腰3+肩4+肘2)
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()

        default_angles_full = np.concatenate([
            self.config.default_angles,  # 12
            self.config.arm_waist_target[[0, 1, 2, 3, 4, 6, 10, 11, 13]]  # 9
        ])  # = 21
        qj_obs = (qj_obs - default_angles_full) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        period = 0.7
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        # self.cmd[0] = self.remote_controller.ly
        # self.cmd[1] = self.remote_controller.lx * -1
        # self.cmd[2] = self.remote_controller.rx * -1

        self.update_command()


        # === gait 模式识别（新版，严格对齐 simulation） ===
        lin_vel = np.linalg.norm(self.cmd[:2])  # XY平面速度模长
        vx, vy = self.cmd[0], self.cmd[1]
        froude_number = np.square(self.cmd[0]) / (9.81 * self.leg_length)  # 只用X方向速度计算 Froude number
        # self.running_mask = froude_number > 0.5

        self.standing_command = lin_vel < 0.1
        self.running_mask = froude_number > 0.5
        self.walking_mask = not (self.standing_command or self.running_mask)
        # self.walking_mask = not (self.standing_command)


        # 初始化历史状态
        if not hasattr(self, "prev_walking_mask"):
            self.prev_walking_mask = True
        if not hasattr(self, "prev_running_mask"):
            self.prev_running_mask = False
        if not hasattr(self, "walk_to_stand_transition_timer"):
            self.walk_to_stand_transition_timer = 0.0
        if not hasattr(self, "in_walk_to_stand_transition"):
            self.in_walk_to_stand_transition = False
        if not hasattr(self, "run_to_walk_transition_timer"):
            self.run_to_walk_transition_timer = 0.0
        if not hasattr(self, "in_run_to_walk_transition"):
            self.in_run_to_walk_transition = False

        # 判断是否刚刚从 walk 切换为 stand（用于触发过渡期）
        just_walk_to_stand = self.prev_walking_mask and self.standing_command
        just_run_to_walk  = self.prev_running_mask and self.walking_mask

        # 更新 transition 状态
        if just_walk_to_stand:
            self.walk_to_stand_transition_timer = 0.0
            self.in_walk_to_stand_transition = True
        elif self.standing_command:
            self.walk_to_stand_transition_timer += self.config.control_dt
            if self.walk_to_stand_transition_timer >= 1.5:
                self.in_walk_to_stand_transition = False

        # Step 5: 更新 run→walk
        if just_run_to_walk:
            self.run_to_walk_transition_timer = 0.0
            self.in_run_to_walk_transition = True
        elif self.walking_mask:
            self.run_to_walk_transition_timer += self.config.control_dt
            if self.run_to_walk_transition_timer >= 3.0:
                self.in_run_to_walk_transition = False

        # Step 6: 设置 gait ID
        # 0: Stand, 1: Walk, 2: Walk→Stand, 3: Run, 4: Run→Walk
        if self.running_mask:
            gait_id = 3
        elif self.in_run_to_walk_transition:
            gait_id = 4
        elif self.walking_mask:
            gait_id = 1
        elif self.in_walk_to_stand_transition:
            gait_id = 2
        else:
            gait_id = 0

        # Step 7: 输出 gait 编码
        self.gait_mode = float(gait_id)
        self.gait_mode_onehot = np.zeros(6, dtype=np.float32)
        self.gait_mode_onehot[gait_id] = 1.0
        gait_mode_onehot = self.gait_mode_onehot

        # Step 8: 更新历史状态
        self.prev_walking_mask = self.walking_mask
        self.prev_running_mask = self.running_mask

        # # 构造 observation (86维)
        # obs_list = [
        #     ang_vel[0],                                 # 3
        #     gravity_orientation,                        # 3
        #     self.cmd * self.config.cmd_scale * self.config.max_cmd,  # 3
        #     qj_obs, dqj_obs, self.action,               # 3 * 23
        #     [sin_phase, cos_phase],                     # 2
        #     gait_mode_onehot                            # 6
        # ]

        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        self.obs[9 + num_actions * 3] = sin_phase
        self.obs[9 + num_actions * 3 + 1] = cos_phase

        # 正确插入 gait_mode_onehot（切片方式）
        start_idx = 9 + num_actions * 3 + 2
        self.obs[start_idx : start_idx + 6] = gait_mode_onehot


        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        # target_dof_pos = self.config.default_angles + self.action * self.config.action_scale
        target_dof_pos = default_angles_full + self.action * self.config.action_scale
        # Build low cmd
        # for i in range(len(self.config.leg_joint2motor_idx)):
        #     motor_idx = self.config.leg_joint2motor_idx[i]
        #     self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
        #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
        #     self.low_cmd.motor_cmd[motor_idx].tau = 0

        # for i in range(len(self.config.arm_waist_joint2motor_idx)):
        #     motor_idx = self.config.arm_waist_joint2motor_idx[i]
        #     if i in [3, 10]:  # shoulder_pitch
        #         shoulder_idx = 14 if i == 3 else 15  # 注意，14/15是shoulder_pitch动作
        #         print('i:',i)
        #         print('shoulder_idx:',shoulder_idx)
        #         self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[shoulder_idx]
        #         print('self.low_cmd.motor_cmd[motor_idx].q:',self.low_cmd.motor_cmd[motor_idx].q)
        #         print('target_dof_pos[shoulder_idx]:',target_dof_pos[shoulder_idx])

        #     else:
        #         self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
        #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
        #     self.low_cmd.motor_cmd[motor_idx].tau = 0



        # === 控制策略输出的 21 个关节 ===
        for i, motor_idx in enumerate(self.controlled_joint2motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.controlled_kps[i]  # ✅ 修改这里
            self.low_cmd.motor_cmd[motor_idx].kd = self.controlled_kds[i]  # ✅
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        # === 其余非策略控制的 arm_waist 关节，用 default 固定 ===
        controlled_motor_set = set(self.controlled_joint2motor_idx)
        for i, motor_idx in enumerate(self.config.arm_waist_joint2motor_idx):
            if motor_idx not in controlled_motor_set:
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)
        # === 日志记录（记录膝盖和肩关节） ===
        left_knee_idx = self.config.leg_joint2motor_idx[3]
        right_knee_idx = self.config.leg_joint2motor_idx[9]
        left_elbow_idx = self.config.arm_waist_joint2motor_idx[6]
        right_elbow_idx = self.config.arm_waist_joint2motor_idx[13]
        imu = self.low_state.imu_state
        quat = imu.quaternion
        ang_vel = imu.gyroscope
        acc = imu.accelerometer
        rpy = imu.rpy
        log_entry = {
            "time": self.counter * self.config.control_dt,
            "gait_id": gait_id,
            "cmd_forward": self.cmd[0],
            "cmd_lateral": self.cmd[1],
            "cmd_turn": self.cmd[2],
            "action": self.action.copy(),

            "imu_quat_w": quat[0],
            "imu_quat_x": quat[1],
            "imu_quat_y": quat[2],
            "imu_quat_z": quat[3],
            "imu_gyro_x": ang_vel[0],
            "imu_gyro_y": ang_vel[1],
            "imu_gyro_z": ang_vel[2],
            "imu_acc_x": acc[0],
            "imu_acc_y": acc[1],
            "imu_acc_z": acc[2],
            "imu_roll": rpy[0],
            "imu_pitch": rpy[1],
            "imu_yaw": rpy[2],

            # === Knee joints
            "left_knee": target_dof_pos[3],
            "right_knee": target_dof_pos[9],
            "left_knee_actual": self.low_state.motor_state[left_knee_idx].q,
            "right_knee_actual": self.low_state.motor_state[right_knee_idx].q,

            # === Shoulder pitch joints
            "left_shoulder_target": target_dof_pos[15],
            "right_shoulder_target": target_dof_pos[18],
            "left_shoulder_pitch_actual": self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[3]].q,
            "right_shoulder_pitch_actual": self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[10]].q,

            # === Elbow joints
            "left_elbow_target": target_dof_pos[17],
            "right_elbow_target": target_dof_pos[20],
            "left_elbow_actual": self.low_state.motor_state[left_elbow_idx].q,
            "right_elbow_actual": self.low_state.motor_state[right_elbow_idx].q,

            # === (Optional) Heuristic contact estimate
            "left_contact_est": int(abs(self.dqj[3]) < 0.15 and self.qj[3] > 0.4),
            "right_contact_est": int(abs(self.dqj[9]) < 0.15 and self.qj[9] > 0.4),
        }
        for i, motor in enumerate(self.low_state.motor_state):
            name = self.joint_name_by_motor.get(i, f"motor_{i}")
            log_entry[f"{name}_pos"] = motor.q
            log_entry[f"{name}_vel"] = motor.dq
            log_entry[f"{name}_acc"] = motor.ddq
            log_entry[f"{name}_tau_est"] = motor.tau_est

        self.log_data.append(log_entry)
        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()


    print("Launching deployment...")

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)
    print("Launching deployment...")

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)
    print("Launching deployment...")


    controller = Controller(config)
    print("!!!!!!!!!Launching deployment...")


    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()
    print("Launching deployment...")


    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()
    print("Launching deployment...")
    print(f"Using interface: {sys.argv[1]}, config: {sys.argv[2]}")

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    save_path = f"/home/tianhu/unitree_rl_gym/data/g1_real/knee_shoulder_log_{int(time.time())}.csv"
    df = pd.DataFrame(controller.log_data)
    df.to_csv(save_path, index=False)
    print(f"✅ 膝盖与肩关节数据已保存至: {save_path}")
    print("Exit")