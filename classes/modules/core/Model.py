import math
import os
from typing import Union

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn.functional import normalize
from torchvision.transforms import transforms

from auxiliary.settings import DEVICE
from auxiliary.utils import correct, rescale, scale


class Model:

    def __init__(self):
        self._device = DEVICE
        self._network = None
        self.__optimizer = None

    def predict(self, x: torch.Tensor, m: torch.Tensor = None) -> Union[torch.Tensor, tuple]:
        pass

    def print_network(self):
        print(self._network)

    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network.txt"), 'a+').write(str(self._network))

    def train_mode(self):
        self._network = self._network.train()

    def evaluation_mode(self):
        self._network = self._network.eval()

    def save(self, path_to_file: str):
        torch.save(self._network.state_dict(), path_to_file)

    def load(self, path_to_pretrained: str):
        self._network.load_state_dict(torch.load(path_to_pretrained, map_location=self._device))

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "rmsprop"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
        self.__optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate)

    def optimize(self, x: Tensor, y: Tensor, m: Tensor = None) -> float:
        self.__optimizer.zero_grad()
        pred = self.predict(x, m)
        loss = self.get_angular_loss(pred, y)
        loss.backward()
        self.__optimizer.step()
        return loss.item()

    def get_loss(self, x: Tensor, y: Tensor) -> float:
        return self.get_angular_loss(x, y).item()

    @staticmethod
    def get_angular_loss(pred: Tensor, y: Tensor, safe_v: float = 0.999999) -> Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(y, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)

    @staticmethod
    def get_total_variation_loss(x: Tensor, alpha: float = 0.00001) -> Tensor:
        """
        Computes the total variation regularization (anisotropic version)
        -> Reference: https://www.wikiwand.com/en/Total_variation_denoising

        The total variation regularization of the learnable attention mask encourages spatial smoothness of the mask
        """
        diff_i = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        return alpha * (diff_i + diff_j)

    def get_reg_angular_loss(self, pred, y, attention_mask) -> torch.Tensor:
        angular_loss = self.get_angular_loss(pred, y)
        total_variation_loss = self.get_total_variation_loss(attention_mask)
        return angular_loss + total_variation_loss

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
