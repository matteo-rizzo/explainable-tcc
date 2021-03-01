from typing import Tuple, Union

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

    def compute_loss(self, sequence: Tensor, label: Tensor, mimic: Tensor = None) -> float:
        pred = self.predict(sequence, mimic)
        loss = self.get_angular_loss(pred, label)
        loss.backward()
        return loss.item()
