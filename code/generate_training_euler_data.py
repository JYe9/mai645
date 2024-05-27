import read_bvh
import numpy as np
from os import listdir
import os

def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    print("Generating Euler training data for " + src_bvh_folder)
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
    bvh_dances_names = listdir(src_bvh_folder)
    for bvh_dance_name in bvh_dances_names:
        if bvh_dance_name.endswith(".bvh"):
            bvh_data = read_bvh.parse_frames(src_bvh_folder + bvh_dance_name)
            position_data = bvh_data[:, :3]  # Extract ZPosition, YPosition, XPosition columns
            euler_data = bvh_data[:, 3:]  # Extract Zrotation, Yrotation, Xrotation columns
            normalized_euler_data = (euler_data + 180) / 360  # Normalize to [0, 1] range
            normalized_data = np.concatenate((position_data, normalized_euler_data), axis=1)
            np.save(tar_traindata_folder + bvh_dance_name + ".npy", normalized_data)

def generate_bvh_from_euler_traindata(src_train_folder, tar_bvh_folder):
    print("Generating bvh data for " + src_train_folder)
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)
    dances_names = listdir(src_train_folder)
    for dance_name in dances_names:
        if dance_name.endswith(".npy"):
            normalized_data = np.load(src_train_folder + dance_name)
            position_data = normalized_data[:, :3]  # Extract ZPosition, YPosition, XPosition columns
            normalized_euler_data = normalized_data[:, 3:]  # Extract normalized Euler angle columns
            euler_data = normalized_euler_data * 360 - 180  # Denormalize to original range
            bvh_data = np.concatenate((position_data, euler_data), axis=1)
            read_bvh.write_frames(standard_bvh_file, tar_bvh_folder + dance_name + ".bvh", bvh_data)

standard_bvh_file = "../train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)
print('skeleton: ', skeleton)

# Encode data from bvh to positional encoding
generate_euler_traindata_from_bvh("../train_data_bvh/martial/", "../train_data_euler/martial/")

# Decode from positional to bvh
generate_bvh_from_euler_traindata("../train_data_euler/martial/", "../test_data_euler_bvh/martial/")