from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.functional import normalize

from auxiliary.utils import scale
from classes.modules.common.BaseTCCNet import BaseTCCNet
from classes.modules.common.ConfidenceFCN import ConfidenceFCN

""" Confidence as spatial attention + Confidence as temporal attention """


class ConfTCCNet(BaseTCCNet):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5):
        super().__init__(hidden_size, kernel_size)
        self.fcn = ConfidenceFCN()

    def forward(self, x: torch.Tensor) -> Tuple:
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)

        # Spatial confidence
        rgb, spat_confidence = self.fcn(x)
        spat_weighted_est = scale(rgb * spat_confidence).clone()

        # Temporal confidence
        temp_confidence = F.softmax(torch.mean(torch.mean(spat_confidence.squeeze(1), dim=1), dim=1))
        spat_temp_weighted_est = spat_weighted_est * temp_confidence

        _, _, h, w = spat_weighted_est.shape
        self.conv_lstm.init_hidden(self.hidden_size, (h, w))
        hidden, cell = self.__init_hidden(batch_size, h, w)

        hidden_states = []
        for t in range(time_steps):
            hidden, cell = self.conv_lstm(spat_temp_weighted_est[t, :, :, :, :], hidden, cell)
            hidden_states.append(hidden)

        y = self.fc(torch.mean(torch.stack(hidden_states), dim=0))
        pred = normalize(torch.sum(torch.sum(y, 2), 2), dim=1)
        return pred, spat_confidence, temp_confidence
