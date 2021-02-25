import torch

from classes.modules.common.BaseModel import BaseModel
from classes.modules.multiframe.attention_tccnet.AttentionTCCNet import AttentionTCCNet


class ModelAttentionTCCNet(BaseModel):

    def __init__(self):
        super().__init__()
        self._network = AttentionTCCNet().float().to(self._device)

    def predict(self, sequence: torch.Tensor, mimic: torch.Tensor = None) -> torch.Tensor:
        return self._network(sequence, mimic)

    def compute_loss(self, sequence: torch.Tensor, label: torch.Tensor, mimic: torch.Tensor = None) -> float:
        pred = self.predict(sequence, mimic)
        loss = self.get_angular_loss(pred, label)
        loss.backward()
        return loss.item()
