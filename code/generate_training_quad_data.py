import read_bvh
import numpy as np
from os import listdir
import os
import transforms3d.euler as euler
import transforms3d.quaternions as quat

def euler_to_quaternion(euler_angles):
    # Convert euler angles to quaternion
    return quat.quat2mat(euler.euler2quat(euler_angles[2], euler_angles[1], euler_angles[0], 'rzyx'))

def quaternion_to_euler(quaternion):
    # Convert quaternion to euler angles
    return euler.quat2euler(quat.mat2quat(quaternion), 'rzyx')

def generate_quad_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    # TODO:
    print("Generating quaternions training data for " + src_bvh_folder)
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
    bvh_dances_names = listdir(src_bvh_folder)
    for bvh_dance_name in bvh_dances_names:
        name_len = len(bvh_dance_name)
        if name_len > 4:
            if bvh_dance_name[name_len - 4:] == ".bvh":
                bvh_data = read_bvh.parse_frames(src_bvh_folder + bvh_dance_name)
                euler_data = bvh_data[:, 3:]
                quad_data = []
                for frame in euler_data:
                    euler_angles = frame.reshape(-1, 3)
                    quad_frame = []
                    for joint_angles in euler_angles:
                        quad = euler_to_quaternion(joint_angles)
                        quad_frame.append(quad.flatten())
                    quad_data.append(np.concatenate(quad_frame))
                np.save(tar_traindata_folder + bvh_dance_name + ".npy", np.array(quad_data))

def generate_bvh_from_quad_traindata(src_train_folder, tar_bvh_folder):
    # TODO:
    print("Generating bvh data from quaternions training data for " + src_train_folder)
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)
    dances_names = listdir(src_train_folder)
    for dance_name in dances_names:
        name_len = len(dance_name)
        if name_len > 4:
            if dance_name[name_len - 4:] == ".npy":
                quad_data = np.load(src_train_folder + dance_name)
                num_frames = quad_data.shape[0]
                num_joints = int(quad_data.shape[1] / 9)
                bvh_data = np.zeros((num_frames, num_joints * 3 + 6))
                for i in range(num_frames):
                    quad_frame = quad_data[i]
                    euler_frame = []
                    for j in range(num_joints):
                        quad = quad_frame[j * 9:(j + 1) * 9].reshape(3, 3)
                        euler_angles = quaternion_to_euler(quad)
                        euler_frame.append(euler_angles)
                    bvh_data[i, 6:] = np.concatenate(euler_frame)
                read_bvh.write_frames(standard_bvh_file, tar_bvh_folder + dance_name + ".bvh", bvh_data)

standard_bvh_file = "../train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

# Encode data from bvh to positional encoding
generate_quad_traindata_from_bvh("../train_data_bvh/martial/","../train_data_quad/martial/")
# generate_quad_traindata_from_bvh("../train_data_bvh/indian/","../train_data_quad/indian/")
# generate_quad_traindata_from_bvh("../train_data_bvh/salsa/","../train_data_quad/salsa/")

# Decode from positional to bvh
generate_bvh_from_quad_traindata("../train_data_quad/martial/", "../test_data_quad_bvh/martial/",)
# generate_bvh_from_quad_traindata("../train_data_quad/indian/", "../test_data_quad_bvh/indian/",)
# generate_bvh_from_quad_traindata("../train_data_quad/salsa/", "../test_data_quad_bvh/salsa/",)
