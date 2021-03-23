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
