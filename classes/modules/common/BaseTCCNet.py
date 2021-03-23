from typing import Tuple

import torch
from torch import nn

from classes.modules.common.conv_lstm.ConvLSTMCell import ConvLSTMCell
from settings import DEVICE


class BaseTCCNet(nn.Module):

    def __init__(self, rnn_input_size: int = 3, hidden_size: int = 128, kernel_size: int = 3):
        super().__init__()
        self.device = DEVICE
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        # Recurrent component for aggregating spatial encodings
        self.conv_lstm = ConvLSTMCell(rnn_input_size, hidden_size, kernel_size)

        # Final classifier
        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=6, stride=1, padding=3),
            nn.Sigmoid(),
            nn.Conv2d(hidden_size, 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size: int, h: int, w: int) -> Tuple:
        hidden_state = torch.zeros((batch_size, self.hidden_size, h, w)).to(self.device)
        cell_state = torch.zeros((batch_size, self.hidden_size, h, w)).to(self.device)
        return hidden_state, cell_state
