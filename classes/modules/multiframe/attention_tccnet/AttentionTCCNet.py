from typing import Tuple

import torch
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

        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(self.hidden_size * 2, self.hidden_size // 2, kernel_size=6, stride=1, padding=3),
            nn.Sigmoid(),
            nn.Conv2d(self.hidden_size // 2, 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def __init_hidden(self, batch_size: int, h: int, w: int) -> Tuple:
        hidden_state = torch.zeros((batch_size, self.hidden_size, h, w)).to(self.device)
        cell_state = torch.zeros((batch_size, self.hidden_size, h, w)).to(self.device)
        return hidden_state, cell_state

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        @param a: the sequences of frames of shape "bs x ts x nc x h x w"
        @param b: the mimic sequences of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant prediction
        """

        batch_size, time_steps, num_channels, h, w = a.shape

        a = a.view(batch_size * time_steps, num_channels, h, w)
        b = b.view(batch_size * time_steps, num_channels, h, w)

        _, rgb_a, confidence_a = self.fc4_A(a)
        _, rgb_b, confidence_b = self.fc4_B(b)

        weighted_est_a = scale(rgb_a * confidence_a).clone()
        weighted_est_b = scale(rgb_b * confidence_b).clone()

        _, _, h_a, w_a = weighted_est_a.shape
        self.lstm_A.init_hidden(self.hidden_size, (h_a, w_a))
        hidden_1, cell_1 = self.__init_hidden(batch_size, h_a, w_a)

        _, _, h_b, w_b = weighted_est_b.shape
        self.lstm_B.init_hidden(self.hidden_size, (h_b, w_b))
        hidden_2, cell_2 = self.__init_hidden(batch_size, h_b, w_b)

        for t in range(time_steps):
            hidden_1, cell_1 = self.lstm_A(weighted_est_a[t, :, :, :].unsqueeze(0), hidden_1, cell_1)
            hidden_2, cell_2 = self.lstm_B(weighted_est_b[t, :, :, :].unsqueeze(0), hidden_2, cell_2)

        c = torch.cat((hidden_1, hidden_2), 1)
        c = self.fc(c)

        pred = normalize(torch.sum(torch.sum(c, 2), 2), dim=1)

        return pred, rgb_a, confidence_a
