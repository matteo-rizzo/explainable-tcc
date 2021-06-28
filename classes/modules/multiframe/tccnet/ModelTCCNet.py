import torch

from classes.modules.core.BaseModel import BaseModel
from classes.modules.multiframe.tccnet.TCCNet import TCCNet


class ModelTCCNet(BaseModel):

    def __init__(self, use_shot_branch: bool):
        super().__init__()
        self._network = TCCNet(use_shot_branch).to(self._device)

    def predict(self, sequence: torch.Tensor, mimic: torch.Tensor = None) -> torch.Tensor:
        return self._network(sequence, mimic)

    def compute_loss(self, sequence: torch.Tensor, label: torch.Tensor, mimic: torch.Tensor = None) -> float:
        pred = self.predict(sequence, mimic)
        loss = self.get_angular_loss(pred, label)
        loss.backward()
        return loss.item()
