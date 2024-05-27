import read_bvh
import numpy as np
from os import listdir
import os
from scipy.spatial.transform import Rotation as R

def R_x(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    zeros = np.zeros_like(theta)
    ones = np.ones_like(theta)
    return np.stack([
        np.stack([ones, zeros, zeros], axis=-1),
        np.stack([zeros, cos_theta, -sin_theta], axis=-1),
        np.stack([zeros, sin_theta, cos_theta], axis=-1)
    ], axis=-2)

def R_y(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    zeros = np.zeros_like(theta)
    ones = np.ones_like(theta)
    return np.stack([
        np.stack([cos_theta, zeros, sin_theta], axis=-1),
        np.stack([zeros, ones, zeros], axis=-1),
        np.stack([-sin_theta, zeros, cos_theta], axis=-1)
    ], axis=-2)

def R_z(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    zeros = np.zeros_like(theta)
    ones = np.ones_like(theta)
    return np.stack([
        np.stack([cos_theta, -sin_theta, zeros], axis=-1),
        np.stack([sin_theta, cos_theta, zeros], axis=-1),
        np.stack([zeros, zeros, ones], axis=-1)
    ], axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converting a rotation matrix to a 6D rotation representation
    Args.
        matrix: rotation matrix, shape (*, 3, 3)
    Returns.
        6D rotation matrix, shape (*, 6)
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converting a 6D rotation representation to a rotation matrix
    Args.
        d6: 6D rotation matrix with shape (*, 6).
    Returns.
        Rotated matrix, shape (*, 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def get_6D_rotation(euler_angles):
    """
    Converting Euler angles to 6D rotational representation
    Args.
        euler_angles: Euler angles, shape (*, 3).
    Returns.
        6D rotated representation, shape (*, 6)
    """
    R_x_val = R_x(euler_angles[..., 0])
    R_y_val = R_y(euler_angles[..., 1])
    R_z_val = R_z(euler_angles[..., 2])

    R = np.einsum('...ij,...jk->...ik', R_z_val, np.einsum('...ij,...jk->...ik', R_y_val, R_x_val))

    return matrix_to_rotation_6d(R)


def get_euler_angles(rotation_6d):
    """
    Converting 6D rotation representations to Euler angles
    Args.
        rotation_6d: 6D rotation, shape (*, 6).
    Returns.
        Euler angle, shape (*, 3).
    """
    rotation_matrices = rotation_6d_to_matrix(rotation_6d)
    euler_angles = R.from_matrix(rotation_matrices).as_euler('XYZ', degrees=True)
    return euler_angles


def generate_6D_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    print("Generating 6D training data for " + src_bvh_folder)
    if (os.path.exists(tar_traindata_folder) == False):
        os.makedirs(tar_traindata_folder)

    bvh_files = listdir(src_bvh_folder)
    for bvh_file in bvh_files:
        if bvh_file.endswith(".bvh"):
            bvh_path = os.path.join(src_bvh_folder, bvh_file)

            dance = read_bvh.parse_frames(bvh_path)
            euler_angles = dance[:, 3:]

            rotation_6d = get_6D_rotation(euler_angles)

            save_path = os.path.join(tar_traindata_folder, bvh_file + ".npy")
            np.save(save_path, rotation_6d)


def generate_bvh_from_6D_traindata(src_train_folder, tar_bvh_folder):
    print("Generating bvh data from 6D training data for " + src_train_folder)
    if (os.path.exists(tar_bvh_folder)==False):
        os.makedirs(tar_bvh_folder)

    train_files = listdir(src_train_folder)
    for train_file in train_files:
        if train_file.endswith(".npy"):
            train_path = os.path.join(src_train_folder, train_file)

            rotation_6d = np.load(train_path)
            euler_angles = get_euler_angles(rotation_6d)

            num_frames = euler_angles.shape[0]
            num_joints = euler_angles.shape[1] // 3

            bvh_data = np.zeros((num_frames, num_joints * 3 + 6))
            bvh_data[:, 3:6] = rotation_6d[:, :3]  # Set hip rotation to the first 3 columns of 6D representation

            for i in range(num_joints):
                bvh_data[:, 6 + i * 3: 6 + (i + 1) * 3] = euler_angles[:, i * 3: (i + 1) * 3]

            save_path = os.path.join(tar_bvh_folder, train_file[:-4] + ".bvh")
            read_bvh.write_frames(standard_bvh_file, save_path, bvh_data)


standard_bvh_file = "../train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

# Encode data from bvh to 6D representation
generate_6D_traindata_from_bvh("../train_data_bvh/martial/", "../train_data_6D/martial/")

# Decode from 6D representation to bvh
generate_bvh_from_6D_traindata("../train_data_6D/martial/", "../test_data_6D_bvh/martial/")