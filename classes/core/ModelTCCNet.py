from typing import Union, Tuple

import torch
from torch import Tensor

from classes.core.Model import Model
from classes.losses.AngularLoss import AngularLoss


class ModelTCCNet(Model):

    def __init__(self):
        super().__init__()
        self._criterion = AngularLoss(self._device)

    def get_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        return self._criterion(pred, label)

    def predict(self, x: torch.Tensor, m: torch.Tensor) -> Union[torch.Tensor, Tuple]:
        pass

    def optimize(self, x: Tensor, y: Tensor, m: Tensor = None) -> float:
        self._optimizer.zero_grad()
        pred = self.predict(x, m)
        loss = self.get_loss(pred, y)
        loss.backward()
        self._optimizer.step()
        return loss.item()
