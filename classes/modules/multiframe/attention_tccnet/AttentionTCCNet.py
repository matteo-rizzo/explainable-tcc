from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.functional import normalize

from auxiliary.settings import DEVICE
from auxiliary.utils import scale
from classes.modules.common.conv_lstm.ConvLSTMCell import ConvLSTMCell
from classes.modules.multiframe.attention_tccnet.submodules.FC4 import FC4

"""
TCCNet presented in 'A Benchmark for Temporal Color Constancy' <https://arxiv.org/abs/2003.03763>
Refer to <https://github.com/yanlinqian/Temporal-Color-Constancy> for the original implementation
"""


class AttentionTCCNet(nn.Module):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5):
        super().__init__()

        self.device = DEVICE
        self.hidden_size = hidden_size

        self.fc4_A = FC4()
        self.fc4_B = FC4()

        self.lstm_A = ConvLSTMCell(3, self.hidden_size, kernel_size)
        self.lstm_B = ConvLSTMCell(3, self.hidden_size, kernel_size)

        self.phi_h = nn.Linear(hidden_size, 1, bias=False)
        self.phi_x = nn.Linear(3, 1, bias=False)

        # self.fc = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #     nn.Conv2d(self.hidden_size * 2, self.hidden_size // 2, kernel_size=6, stride=1, padding=3),
        #     nn.Sigmoid(),
        #     nn.Conv2d(self.hidden_size // 2, 3, kernel_size=1, stride=1),
        #     nn.Sigmoid()
        # )

        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=6, stride=1, padding=3),
            nn.Sigmoid(),
            nn.Conv2d(self.hidden_size, 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def __init_hidden(self, batch_size: int, h: int, w: int) -> Tuple:
        hidden_state = torch.zeros((batch_size, self.hidden_size, h, w)).to(self.device)
        cell_state = torch.zeros((batch_size, self.hidden_size, h, w)).to(self.device)
        return hidden_state, cell_state

    def __generate_energy(self, masked_features: torch.Tensor, hidden: torch.Tensor) -> np.ndarray:
        """
        Computes energy as e_ti = phi(H_t-1, masked_X_i) = phi(H_t-1) + phi(masked_X_i)
        @param masked_features: the i-th masked spatial features map
        @param hidden: the hidden states of the RNN at time step t-1
        @return: the energy for the i-th attended frame at time step t-1,
        """
        att_x = self.phi_x(torch.mean(torch.mean(masked_features, dim=2), dim=1))
        att_h = self.phi_h(torch.mean(torch.mean(hidden, dim=3), dim=2))
        e = att_x + att_h
        return e.squeeze(dim=1).cpu().data.numpy()

    @staticmethod
    def __energies_to_weights(energies: List) -> torch.Tensor:
        """
        Applies softmax to energies to scale them in [0, 1]
        @param energies: a list of energies e_ti (of length time steps), one for each attended frame
        @return: the weights w_ti = exp(e_ti)/sum_i^n(exp(e_ti))
        """
        weights = F.softmax(torch.from_numpy(np.asarray(energies).squeeze()), dim=0).to(DEVICE)
        return weights.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    #     """
    #     @param a: the sequences of frames of shape "bs x ts x nc x h x w"
    #     @param b: the mimic sequences of shape "bs x ts x nc x h x w"
    #     @return: the normalized illuminant prediction
    #     """
    #     batch_size, time_steps, num_channels, h, w = a.shape
    #
    #     a = a.view(batch_size * time_steps, num_channels, h, w)
    #     b = b.view(batch_size * time_steps, num_channels, h, w)
    #
    #     _, rgb_a, confidence_a = self.fc4_A(a)
    #     _, rgb_b, confidence_b = self.fc4_B(b)
    #
    #     weighted_est_a = scale(rgb_a * confidence_a).clone()
    #     weighted_est_b = scale(rgb_b * confidence_b).clone()
    #
    #     _, _, h_a, w_a = weighted_est_a.shape
    #     self.lstm_A.init_hidden(self.hidden_size, (h_a, w_a))
    #     hidden_1, cell_1 = self.__init_hidden(batch_size, h_a, w_a)
    #
    #     _, _, h_b, w_b = weighted_est_b.shape
    #     self.lstm_B.init_hidden(self.hidden_size, (h_b, w_b))
    #     hidden_2, cell_2 = self.__init_hidden(batch_size, h_b, w_b)
    #
    #     for t in range(time_steps):
    #         hidden_1, cell_1 = self.lstm_A(weighted_est_a[t, :, :, :].unsqueeze(0), hidden_1, cell_1)
    #         hidden_2, cell_2 = self.lstm_B(weighted_est_b[t, :, :, :].unsqueeze(0), hidden_2, cell_2)
    #
    #     c = torch.cat((hidden_1, hidden_2), 1)
    #     c = self.fc(c)
    #
    #     pred = normalize(torch.sum(torch.sum(c, 2), 2), dim=1)
    #
    #     return pred, rgb_a, confidence_a

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        @param a: the sequences of frames of shape "bs x ts x nc x h x w"
        @param b: the mimic sequences of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant prediction
        """
        batch_size, time_steps, num_channels, h, w = a.shape

        a = a.view(batch_size * time_steps, num_channels, h, w)

        _, rgb_a, confidence_a = self.fc4_A(a)
        s_weighted_est_a = scale(rgb_a * confidence_a).clone()

        _, _, h_a, w_a = s_weighted_est_a.shape
        self.lstm_A.init_hidden(self.hidden_size, (h_a, w_a))
        hidden_1, cell_1 = self.__init_hidden(batch_size, h_a, w_a)

        hidden_states = []
        for _ in range(time_steps):
            # Compute energies as e_ti = phi(H_t-1, masked_X_i)
            energies = [self.__generate_energy(s_weighted_est_a[i, :, :, :], hidden_1) for i in range(time_steps)]

            # Energies to temporal weights via softmax: w_ti = exp(e_ti)/sum_i^n(exp(e_ti))
            weights = self.__energies_to_weights(energies)

            # Final feature map as weighted sum of features from all frames: Y_t = 1/n sum_i^n(w_ti masked_X_i)
            t_weighted_est_a = torch.div(torch.sum(s_weighted_est_a * weights, dim=0), time_steps)

            hidden_1, cell_1 = self.lstm_A(t_weighted_est_a.unsqueeze(0), hidden_1, cell_1)

            hidden_states.append(hidden_1)

        y = self.fc(torch.mean(torch.stack(hidden_states), dim=0))
        pred = normalize(torch.sum(torch.sum(y, 2), 2), dim=1)
        return pred, rgb_a, confidence_a
