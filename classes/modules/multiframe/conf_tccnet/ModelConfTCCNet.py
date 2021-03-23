import os
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import transforms

from auxiliary.settings import DEVICE
from auxiliary.utils import correct, rescale, scale
from classes.modules.common.BaseModel import BaseModel
from multiframe.conf_tccnet.ConfTCCNet import ConfTCCNet


class ModelConfTCCNet(BaseModel):

    def __init__(self):
        super().__init__()
        self._network = ConfTCCNet().float().to(self._device)

    def predict(self, sequence: Tensor, m: Tensor = None, return_steps: bool = False) -> Union[Tuple, Tensor]:
        pred, rgb, confidence = self._network(sequence)
        if return_steps:
            return pred, rgb, confidence
        return pred

    def vis_confidence(self, model_output: dict, path_to_plot: str):
        model_output = {k: v.clone().detach().to(DEVICE) for k, v in model_output.items()}

        x, y, pred = model_output["x"], model_output["y"], model_output["pred"]
        rgb, c = model_output["rgb"], model_output["c"]

        original = transforms.ToPILImage()(x.squeeze()).convert("RGB")
        est_corrected = correct(original, pred)

        size = original.size[::-1]
        weighted_est = rescale(scale(rgb * c), size).squeeze().permute(1, 2, 0)
        rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
        c = rescale(c, size).squeeze(0).permute(1, 2, 0)
        masked_original = scale(F.to_tensor(original).to(DEVICE).permute(1, 2, 0) * c)

        plots = [(original, "original"), (masked_original, "masked_original"), (est_corrected, "correction"),
                 (rgb, "per_patch_estimate"), (c, "confidence"), (weighted_est, "weighted_estimate")]

        stages, axs = plt.subplots(2, 3)
        for i in range(2):
            for j in range(3):
                plot, text = plots[i * 3 + j]
                if isinstance(plot, Tensor):
                    plot = plot.cpu()
                axs[i, j].imshow(plot, cmap="gray" if "confidence" in text else None)
                axs[i, j].set_title(text)
                axs[i, j].axis("off")

        os.makedirs(os.sep.join(path_to_plot.split(os.sep)[:-1]), exist_ok=True)
        epoch, loss = path_to_plot.split(os.sep)[-1].split("_")[-1].split(".")[0], self.get_angular_loss(pred, y)
        stages.suptitle("EPOCH {} - ERROR: {:.4f}".format(epoch, loss))
        stages.savefig(os.path.join(path_to_plot), bbox_inches='tight', dpi=200)
        plt.clf()
        plt.close('all')

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

    def get_regularized_loss(self, pred, y, attention_mask) -> torch.Tensor:
        angular_loss = self.get_angular_loss(pred, y)
        total_variation_loss = self.get_total_variation_loss(attention_mask)
        # return angular_loss + total_variation_loss
        return angular_loss

    def compute_loss(self, sequence: Tensor, y: Tensor, m: Tensor = None) -> float:
        pred, _, confidence = self.predict(sequence, m, return_steps=True)
        loss = self.get_regularized_loss(pred, y, attention_mask=confidence)
        loss.backward()
        return loss.item()
