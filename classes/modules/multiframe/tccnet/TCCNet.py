from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from auxiliary.settings import DEVICE
from classes.modules.submodules.conv_lstm.ConvLSTMCell import ConvLSTMCell
from classes.modules.submodules.squeezenet.SqueezeNetLoader import SqueezeNetLoader

"""
TCCNet presented in 'A Benchmark for Temporal Color Constancy' <https://arxiv.org/abs/2003.03763>
Refer to <https://github.com/yanlinqian/Temporal-Color-Constancy> for the original implementation
"""


class TCCNet(nn.Module):

    def __init__(self, use_shot_branch, hidden_size: int = 128, kernel_size: int = 5):
        super().__init__()

        self.device = DEVICE
        self._hidden_size = hidden_size
        self.use_shot_branch = use_shot_branch

        s_temp = SqueezeNetLoader(version=1.1).load(pretrained=True)
        self.squeezenet1_1_temp = nn.Sequential(*list(s_temp.children())[0][:12])
        self.lstm_temp = ConvLSTMCell(512, self._hidden_size, kernel_size)

        if self.use_shot_branch:
            s_shot = SqueezeNetLoader(version=1.1).load(pretrained=True)
            self.squeezenet1_1_shot = nn.Sequential(*list(s_shot.children())[0][:12])
            self.lstm_shot = ConvLSTMCell(512, self._hidden_size, kernel_size)

        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d((self._hidden_size * 2) if self.use_shot_branch else self._hidden_size,
                      (self._hidden_size // 2) if self.use_shot_branch else self._hidden_size,
                      kernel_size=6, stride=1, padding=3),
            nn.Sigmoid(),
            nn.Conv2d((self._hidden_size // 2) if self.use_shot_branch else self._hidden_size,
                      3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def __init_hidden(self, bs: int, h: int, w: int) -> Tuple:
        hidden_state = torch.zeros((bs, self._hidden_size, h, w)).to(self.device)
        cell_state = torch.zeros((bs, self._hidden_size, h, w)).to(self.device)
        return hidden_state, cell_state

    def forward(self, seq_temp: torch.Tensor, seq_shot: torch.Tensor) -> Tuple:
        """
        @param seq_temp: the temporal sequences of frames of shape "bs x ts x nc x h x w"
        @param seq_shot: the shot frame sequences of shape "bs x ts x nc x h x w"
        @return: the illuminant prediction for the shot frame and for each frame in the temporal and shot sequences
        """
        batch_size, time_steps, num_channels, h, w = seq_temp.shape

        seq_temp = seq_temp.view(batch_size * time_steps, num_channels, h, w)
        e_temp = self.squeezenet1_1_temp(seq_temp)
        _, num_channels_temp, h_temp, w_temp = e_temp.shape
        e_temp = e_temp.view(batch_size, time_steps, num_channels_temp, h_temp, w_temp)
        self.lstm_temp.init_hidden(self._hidden_size, (h_temp, w_temp))
        h_temp, c_temp = self.__init_hidden(batch_size, h_temp, w_temp)

        if self.use_shot_branch:
            seq_shot = seq_shot.view(batch_size * time_steps, num_channels, h, w)
            e_shot = self.squeezenet1_1_shot(seq_shot)
            _, num_channels_shot, h_shot, w_shot = e_shot.shape
            e_shot = e_shot.view(batch_size, time_steps, num_channels_shot, h_shot, w_shot)
            self.lstm_shot.init_hidden(self._hidden_size, (h_shot, w_shot))
            h_shot, c_shot = self.__init_hidden(batch_size, h_shot, w_shot)

        for t in range(time_steps):
            h_temp, c_temp = self.lstm_temp(e_temp[:, t, :], h_temp, c_temp)
            if self.use_shot_branch:
                h_shot, c_shot = self.lstm_shot(e_shot[:, t, :], h_shot, c_shot)

        out = torch.cat((h_temp, h_shot), 1) if self.use_shot_branch else h_temp
        out = self.fc(out)
        out = F.normalize(torch.sum(torch.sum(out, 2), 2), dim=1)

        return out
