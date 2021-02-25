import numpy as np
import torch
import torch.cuda
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data
from convlstm import *
from feature_dataloader import *
from torch.autograd import Variable
from utils import *

use_cuda = True


class AttentionTCCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(AttentionTCCNet, self).__init__()
        # attention
        self.att_vw = nn.Linear(49 * 2048, 49, bias=False)
        self.att_hw = nn.Linear(hidden_size, 49, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(49))
        self.att_vw_bn = nn.BatchNorm1d(1)
        self.att_hw_bn = nn.BatchNorm1d(1)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc_attention = nn.Linear(hidden_size, 1)

        self.att_feature_w = nn.Linear(2048, 1, bias=False)
        self.att_hidden_w = nn.Linear(hidden_size, 1, bias=False)
        self.fc1 = nn.Linear(2048, output_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.fc_c0_0 = nn.Linear(2048, 1024)
        self.fc_c0_1 = nn.Linear(1024, 512)
        self.fc_h0_0 = nn.Linear(2048, 1024)
        self.fc_h0_1 = nn.Linear(1024, 512)

        self.c0_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.h0_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.input_size = input_size

        self.mask_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),  # (bs*22, 1, 7, 7)
        )

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        self.convlstm_cell = ConvLSTMCell(input_size=(7, 7),
                                          input_dim=2048,
                                          hidden_dim=512,
                                          kernel_size=(3, 3),
                                          bias=True)

        self.dropout_2d = nn.Dropout2d(p=FLAGS.dropout_ratio)

        self.conv_lstm = ConvLSTM(input_size=(7, 7),
                                  input_dim=2048,
                                  hidden_dim=[512],
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=True)

    def get_start_states(self, input_x):

        h0 = torch.mean(input_x, dim=1)

        c0 = torch.mean(input_x, dim=1)

        h0 = self.h0_conv(h0)
        c0 = self.c0_conv(c0)

        return h0, c0

    def temporal_attention_layer(self, features, hiddens):
        """
        : param features: (batch_size, 2048, 7, 7)
        : param hiddens: (batch_size, hidden_dim)
        :return:
        """

        features_tmp = torch.mean(torch.mean(features, dim=3), dim=2)  # [30x2048x7x7]
        hiddens_tmp = torch.mean(torch.mean(hiddens, dim=3), dim=2)  # [30x512x7x7]
        att_fea = self.att_feature_w(features_tmp)
        # att_fea = self.att_vw_bn(att_fea)
        att_h = self.att_hidden_w(hiddens_tmp)
        # att_h = self.att_hw_bn(att_h)
        att_out = att_fea + att_h
        # att_out = att_h

        return att_out

    def forward(self, input_x):

        batch_size = input_x.shape[0]
        seq_len = input_x.shape[2]

        input_x = self.dropout_2d(input_x)
        input_x = input_x.transpose(1, 2).contiguous()

        input_x = input_x.view(-1, 2048, 7, 7)

        mask = self.mask_conv(input_x)
        mask = mask.view(-1, FLAGS.num_segments, 1, 7, 7)

        input_x = input_x.view(-1, FLAGS.num_segments, 2048, 7, 7)

        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        diff_i = torch.sum(torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]))
        diff_j = torch.sum(torch.abs(mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]))

        tv_loss = FLAGS.tv_reg_factor * (diff_i + diff_j)

        mask_A = (mask > 0.5).type(torch.cuda.FloatTensor)
        mask_B = (mask < 0.5).type(torch.cuda.FloatTensor)
        contrast_loss = -(mask * mask_A).mean(0).sum() * FLAGS.constrast_reg_factor * 0.5 + (mask * mask_B).mean(
            0).sum() * FLAGS.constrast_reg_factor * 0.5

        mask_input_x_org = mask * input_x
        # mask_input_x_org =input_x

        h0, c0 = self.get_start_states(mask_input_x_org)

        output_list = []
        temporal_att_weight_list = []

        for i in range(FLAGS.num_segments):
            temporal_att_weight_list = []
            for j in range(FLAGS.num_segments):
                mask_input_x_for_att_per_frame = mask_input_x_org[:, j, :, :, :]
                temporal_att_weight = self.temporal_attention_layer(mask_input_x_for_att_per_frame, h0)

                squeezed_temporal_att_weight = temporal_att_weight.squeeze(dim=1)
                temporal_att_weight_list.append(squeezed_temporal_att_weight.cpu().data.numpy())

                temporal_att_weight = temporal_att_weight.view(-1, 1, 1, 1)

            temporal_att_weight = Variable(torch.from_numpy(np.asarray(temporal_att_weight_list).squeeze())).transpose(
                0, 1).cuda()

            temporal_att_weight = F.softmax(temporal_att_weight, dim=1)  # [30, 50]
            print("temporal_att_weight: ", temporal_att_weight)
            weighted_mask_input_all_frame = torch.sum(
                mask_input_x_org * (temporal_att_weight.view(-1, FLAGS.num_segments, 1, 1, 1)), dim=1)

            h0, c0 = self.convlstm_cell(weighted_mask_input_all_frame, (h0, c0))

            output = torch.mean(torch.mean(h0, dim=3), dim=2)
            # output = self.fc_out(output)

            output_list.append(output)
            temporal_att_weight_list.append(temporal_att_weight)

        # final_temporal_att_weight =torch.mean(torch.stack(temporal_att_weight_list, dim=0), 0)

        # print("final_temporal_att_weight", final_temporal_att_weight)
        output = torch.mean(torch.stack(output_list, dim=0), 0)
        final_output = self.fc_out(output)
        # final_output = torch.mean(torch.stack(output_list, dim=0),0)

        return final_output, temporal_att_weight, mask, tv_loss, contrast_loss

    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
