import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh

Hip_index = read_bvh.joint_index['hip']

Seq_len = 100
Hidden_size = 1024
# Joints_num = 57
Joints_num = 43
Condition_num = 5
Groundtruth_num = 5
In_frame_size = Joints_num * 3 + 3  # Euler angles plus hip position


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=In_frame_size, hidden_size=1024, out_frame_size=In_frame_size):
        super(acLSTM, self).__init__()

        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        ##lstm#########################################################
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)  # param+ID
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)

    def init_hidden(self, batch):
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        c1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        return ([h0, h1, h2], [c0, c1, c2])

    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))

        out_frame = self.decoder(vec_h2)
        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]

        return (out_frame, vec_h_new, vec_c_new)

    def get_condition_lst(self, condition_num, groundtruth_num, seq_len):
        gt_lst = np.ones((100, groundtruth_num))
        con_lst = np.zeros((100, condition_num))
        lst = np.concatenate((gt_lst, con_lst), 1).reshape(-1)
        return lst[0:seq_len]

    def forward(self, real_seq, condition_num=5, groundtruth_num=5):
        batch = real_seq.size()[0]
        seq_len = real_seq.size()[1]

        condition_lst = self.get_condition_lst(condition_num, groundtruth_num, seq_len)

        (vec_h, vec_c) = self.init_hidden(batch)

        out_seq = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, 1))).cuda())
        out_frame = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.out_frame_size))).cuda())

        for i in range(seq_len):
            if condition_lst[i] == 1:  # input groundtruth frame
                in_frame = real_seq[:, i]
            else:
                in_frame = out_frame

            (out_frame, vec_h, vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)

            out_seq = torch.cat((out_seq, out_frame), 1)

        return out_seq[:, 1: out_seq.size()[1]]

    def calculate_loss(self, out_seq, groundtruth_seq):
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss


def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False,
                       save_bvh_motion=True):

    dif = real_seq_np[:, 1:real_seq_np.shape[1]] - real_seq_np[:, 0: real_seq_np.shape[1]-1]
    real_seq_dif_hip_x_z_np = real_seq_np[:, 0:real_seq_np.shape[1]-1].copy()
    # Replace the values with the difference of each step to the previus step
    real_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
    real_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]

    # check this line to understand the extra steps above
    # print('real vs real diff: ', real_seq_np[0,:5, :10], real_seq_dif_hip_x_z_np[0,:5, :10])

    real_seq = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np.tolist()).cuda())

    seq_len = real_seq.size()[1] - 1
    in_real_seq = real_seq[:, 0:seq_len]
    predict_groundtruth_seq = torch.autograd.Variable(
        torch.FloatTensor(real_seq_dif_hip_x_z_np[:, 1:seq_len + 1].tolist())).cuda().view(real_seq_dif_hip_x_z_np.shape[0], -1)

    # B,SEQ,J
    # print('in real seq: ', in_real_seq.shape)
    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num)
    # print('out fake seq: ', predict_seq.shape)


    optimizer.zero_grad()
    loss = model.calculate_loss(predict_seq, predict_groundtruth_seq)
    loss.backward()
    optimizer.step()

    if print_loss:
        print("###########" + "iter %07d" % iteration + "######################")
        print(loss)
        print("loss: " + str(loss.detach().cpu().numpy()))

    if save_bvh_motion:
        # Save the first motion sequence in the batch
        gt_seq = predict_groundtruth_seq[0].data.cpu().numpy()
        out_seq = predict_seq[0].data.cpu().numpy()

        # print('gt_seq shape: ', gt_seq.shape, Joints_num * 3 + 3)
        # print('out_seq shape: ', out_seq.shape)

        # Calculate the number of frames based on the size of gt_seq and out_seq
        num_frames_gt = gt_seq.size // (Joints_num * 3 + 3)
        num_frames_out = out_seq.size // (Joints_num * 3 + 3)

        # Reshape the sequences to have the correct format
        gt_seq = gt_seq.reshape(num_frames_gt, Joints_num * 3 + 3)
        out_seq = out_seq.reshape(num_frames_out, Joints_num * 3 + 3)

        # print('gt_seq reshape shape: ', gt_seq.shape, num_frames_gt, num_frames_out, out_seq.shape[1])
        # print('out_seq reshape shape: ', out_seq.shape)

        # Reverse the location diff processing
        last_x = 0.0
        last_z = 0.0
        # Change hip xyz previous hip location for ground truth sequence
        for frame in range(gt_seq.shape[0]):
            gt_seq[frame,Hip_index*3]=gt_seq[frame,Hip_index*3]+last_x
            last_x=gt_seq[frame,Hip_index*3]

            gt_seq[frame,Hip_index*3+2]=gt_seq[frame,Hip_index*3+2]+last_z
            last_z=gt_seq[frame,Hip_index*3+2]

        last_x=0.0
        last_z=0.0
        # Change hip xyz based on previous hip locations for out seq
        for frame in range(out_seq.shape[0]):
            out_seq[frame,Hip_index*3]=out_seq[frame,Hip_index*3]+last_x
            last_x=out_seq[frame,Hip_index*3]

            out_seq[frame,Hip_index*3+2]=out_seq[frame,Hip_index*3+2]+last_z
            last_z=out_seq[frame,Hip_index*3+2]


        # print('gt seq and out seq: ', gt_seq.shape, out_seq.shape)

        # Convert the predicted and ground truth sequences to the original BVH format
        # Here use your euler to bvh conversion function
        read_bvh.write_euler_traindata_to_bvh(save_dance_folder + "%07d" % iteration + "_gt.bvh", gt_seq)
        read_bvh.write_euler_traindata_to_bvh(save_dance_folder + "%07d" % iteration + "_out.bvh", out_seq)


def get_dance_len_lst(dances):
    len_lst = []
    for dance in dances:
        length = 10
        if length < 1:
            length = 1
        len_lst = len_lst + [length]

    index_lst = []
    index = 0
    for length in len_lst:
        for i in range(length):
            index_lst = index_lst + [index]
        index = index + 1
    return index_lst


def load_dances(dance_folder):
    dance_files = os.listdir(dance_folder)
    dances = []
    for dance_file in dance_files:
        print("load " + dance_file)
        dance = np.load(dance_folder + dance_file)
        print("frame number: " + str(dance.shape[0]))
        dances = dances + [dance]
    return dances


def train(dances, frame_rate, batch, seq_len, read_weight_path, write_weight_folder,
          write_bvh_motion_folder, in_frame, out_frame, hidden_size=1024, total_iter=500000):
    seq_len = seq_len + 2
    torch.cuda.set_device(0)

    model = acLSTM(in_frame_size=in_frame, hidden_size=hidden_size, out_frame_size=out_frame)

    if read_weight_path != "":
        model.load_state_dict(torch.load(read_weight_path))

    model.cuda()

    current_lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

    model.train()

    dance_len_lst = get_dance_len_lst(dances)
    random_range = len(dance_len_lst)

    speed = frame_rate / 30  # we train the network with frame rate of 30

    for iteration in range(total_iter):
        dance_batch = []
        for b in range(batch):
            dance_id = dance_len_lst[np.random.randint(0, random_range)]
            dance = dances[dance_id].copy()
            dance_len = dance.shape[0]

            start_id = random.randint(10, dance_len - seq_len * speed - 10)
            sample_seq = []
            for i in range(seq_len):
                sample_seq = sample_seq + [dance[int(i * speed + start_id)]]

            T = [0.1 * (random.random() - 0.5), 0.0, 0.1 * (random.random() - 0.5)]
            R = [0, 1, 0, (random.random() - 0.5) * np.pi * 2]
            sample_seq_augmented = read_bvh.augment_train_data(sample_seq, T, R)
            dance_batch = dance_batch + [sample_seq_augmented]

        dance_batch_np = np.array(dance_batch)

        print_loss = False
        save_bvh_motion = False
        if iteration % 1 == 0:
            print_loss = True
        if iteration % 1000 == 0:
            save_bvh_motion = True

        train_one_iteraton(dance_batch_np, model, optimizer, iteration, write_bvh_motion_folder, print_loss,
                           save_bvh_motion)

        if iteration % 1000 == 0:
            path = write_weight_folder + "%07d" % iteration + ".weight"
            torch.save(model.state_dict(), path)


read_weight_path = ""
write_weight_folder = "../train_weight_aclstm_martial_euler/"
write_bvh_motion_folder = "../train_tmp_bvh_aclstm_martial_euler/"
dances_folder = "../train_data_euler/martial/"
dance_frame_rate = 120
batch = 32
in_frame = Joints_num * 3 + 3  # Euler angles plus hip position
out_frame = Joints_num * 3 + 3
hidden_size = 1024

if not os.path.exists(write_weight_folder):
    os.makedirs(write_weight_folder)
if not os.path.exists(write_bvh_motion_folder):
    os.makedirs(write_bvh_motion_folder)

dances = load_dances(dances_folder)

train(dances, dance_frame_rate, batch, 100, read_weight_path, write_weight_folder,
      write_bvh_motion_folder, in_frame, out_frame, hidden_size, total_iter=20001)
# train(dances, dance_frame_rate, batch, 100, read_weight_path, write_weight_folder,
#       write_bvh_motion_folder, in_frame, out_frame, hidden_size, total_iter=20001)