from typing import Tuple
from typing import Union

from torch import Tensor

from classes.modules.common.BaseModel import BaseModel
from classes.modules.multiframe.conf_att_tccnet.ConfAttTCCNet import ConfAttTCCNet


class ModelConfAttTCCNet(BaseModel):

    def __init__(self):
        super().__init__()
        self._network = ConfAttTCCNet().float().to(self._device)

    def predict(self, x: Tensor, m: Tensor = None, return_steps: bool = False) -> Union[Tuple, Tensor]:
        pred, rgb, confidence = self._network(x)
        if return_steps:
            return pred, rgb, confidence
        return pred

    def compute_loss(self, x: Tensor, y: Tensor, m: Tensor = None) -> float:
        pred = self.predict(x, m, return_steps=False)
        loss = self.get_angular_loss(pred, y)
        loss.backward()
        return loss.item()

    def get_loss(self, x: Tensor, y: Tensor) -> float:
        return self.get_angular_loss(x, y).item()
