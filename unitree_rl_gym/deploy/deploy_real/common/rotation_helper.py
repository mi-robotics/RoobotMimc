import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


def transform_imu_data_full(
        waist_roll, waist_roll_omega,
        waist_pitch, waist_pitch_omega,
        waist_yaw, waist_yaw_omega, 
        imu_quat, imu_omega):
    
    def transform(angle, type, w_omega, imu_quat, imu_omega):
        Rw = R.from_euler(type, angle).as_matrix()
        R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
        Rp = np.dot(R_torso, Rw.T)
        w = np.dot(Rw, imu_omega[0]) - np.array([0, 0, w_omega])
        return R.from_matrix(Rp).as_quat()[[3, 0, 1, 2]], w
    
    imu_quat, imu_omega = transform(waist_roll, 'r', waist_roll_omega, imu_quat, imu_omega)    
    imu_quat, imu_omega = transform(waist_pitch, 'p', waist_pitch_omega, imu_quat, imu_omega)
    imu_quat, imu_omega = transform(waist_yaw, 'z', waist_yaw_omega, imu_quat, imu_omega)



def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


def transform_imu_data_from_pelvis_to_torso(
    waist_roll, waist_roll_omega,
    waist_pitch, waist_pitch_omega,
    waist_yaw, waist_yaw_omega,
    pelvis_quat, pelvis_omega
):

    # Convert pelvis quaternion to rotation matrix
    R_pelvis = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    R_waist = R.from_euler('xyz', [waist_roll, waist_pitch, waist_yaw]).as_matrix()
    R_torso = np.dot(R_pelvis, R_waist)

    # --- 2. Calculate Final Angular Velocity ---
    R_roll = R.from_euler('x', waist_roll).as_matrix()
    R_pitch = R.from_euler('y', waist_pitch).as_matrix()

    # Joint velocities in their own local frames
    omega_roll_local = np.array([waist_roll_omega, 0, 0])
    omega_pitch_local = np.array([0, waist_pitch_omega, 0])
    omega_yaw_local = np.array([0, 0, waist_yaw_omega])
    
    omega_roll_in_pelvis = omega_roll_local

    omega_pitch_in_pelvis = np.dot(R_roll, omega_pitch_local)
    # Yaw velocity must be rotated by the preceding roll and pitch
    omega_yaw_in_pelvis = np.dot(np.dot(R_roll, R_pitch), omega_yaw_local)

    # Sum all angular velocities in the pelvis frame
    omega_torso_in_pelvis = pelvis_omega + omega_roll_in_pelvis + omega_pitch_in_pelvis + omega_yaw_in_pelvis
    
    # Finally, transform the total angular velocity from the pelvis frame to the torso's own frame
    # We need the inverse of R_waist (which is its transpose) for this transformation.
    torso_omega = np.dot(R_waist.T, omega_torso_in_pelvis)

    # --- 3. Convert orientation back to quaternion ---
    torso_quat = R.from_matrix(R_torso).as_quat()[[3, 0, 1, 2]]

    return torso_quat, torso_omega


def transform_imu_data_from_pelvis_to_torso_quat(
    waist_roll,
    waist_pitch,
    waist_yaw,
    pelvis_quat
):

    # Convert pelvis quaternion to rotation matrix
    R_pelvis = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    R_waist = R.from_euler('xyz', [waist_roll, waist_pitch, waist_yaw]).as_matrix()
    R_torso = np.dot(R_pelvis, R_waist)
    # --- 3. Conert orientation back to quaternion ---
    torso_quat = R.from_matrix(R_torso).as_quat()[[3, 0, 1, 2]]

    return torso_quat