from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from settings import DEVICE


class TemporalAttentionModule(nn.Module):

    def __init__(self, features_size: int = 512, hidden_size: int = 128):
        super().__init__()
        self.phi_x = nn.Linear(features_size, 1, bias=False)
        self.phi_h = nn.Linear(hidden_size, 1, bias=False)

    def __generate_energy(self, masked_features: Tensor, hidden: Tensor) -> np.ndarray:
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

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        @param x: the sequences of frames of shape "bs x ts x nc x h x w"
        @param h: the hidden state of an RNN
        @return: the normalized illuminant prediction
        """
        time_steps = x.shape[0]

        # Compute energies as e_ti = phi(H_t-1, masked_X_i)
        energies = [self.__generate_energy(x[i, :, :, :], h) for i in range(time_steps)]

        # Energies to temporal weights via softmax: w_ti = exp(e_ti)/sum_i^n(exp(e_ti))
        weights = self.__energies_to_weights(energies)

        return weights
