from typing import Tuple, Union

import torch
from torch import Tensor

from classes.modules.common.BaseModel import BaseModel
from classes.modules.multiframe.attention_tccnet.AttentionTCCNet import AttentionTCCNet


class ModelAttentionTCCNet(BaseModel):

    def __init__(self):
        super().__init__()
        self._network = AttentionTCCNet().float().to(self._device)

    def predict(self, sequence: Tensor, mimic: Tensor = None, return_steps: bool = False) -> Union[Tuple, Tensor]:
        pred, rgb, confidence = self._network(sequence, mimic)
        if return_steps:
            return pred, rgb, confidence
        return pred

    @staticmethod
    def get_total_variation_loss(x: torch.Tensor, reg_factor: float = 0.00001) -> torch.Tensor:
        """
        Computes the total variation regularization (anisotropic version)
        -> Reference: https://www.wikiwand.com/en/Total_variation_denoising

        The total variation regularization of the learnable attention mask encourages spatial smoothness of the mask
        """
        diff_i = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        return reg_factor * (diff_i + diff_j)

    @staticmethod
    def get_contrast_loss(x: torch.Tensor, reg_factor: float = 0.0001) -> torch.Tensor:
        """ The contrast regularization of learnable attention mask is used to suppress the irrelevant information and
         highlight important parts of the input """
        x_a = (x > 0.5).type(torch.FloatTensor)
        x_b = (x < 0.5).type(torch.FloatTensor)
        return -(x * x_a).mean(0).sum() * reg_factor * 0.5 + (x * x_b).mean(0).sum() * reg_factor * 0.5

    def get_regularized_loss(self, pred, label, attention_mask):
        angular_loss = self.get_angular_loss(pred, label)
        total_variation_loss = self.get_total_variation_loss(attention_mask)
        contrast_loss = self.get_contrast_loss(attention_mask)
        return angular_loss + total_variation_loss + contrast_loss

    def compute_loss(self, sequence: Tensor, label: Tensor, mimic: Tensor = None) -> float:
        pred, _, confidence = self.predict(sequence, mimic, return_steps=True)
        loss = self.get_regularized_loss(pred, label, attention_mask=confidence)
        loss.backward()
        return loss.item()
