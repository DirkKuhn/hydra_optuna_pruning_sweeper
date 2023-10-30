import torch as th
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet152


class ResNet(nn.Module):
    def __init__(self, dropout: float, output_dims: list[int], num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        input_dim: int = 28 * 28
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, data: th.Tensor) -> th.Tensor:
        logits = self.layers(data)
        return F.log_softmax(logits, dim=1)


if __name__ == '__main__':
    resnet = resnet50()
    print(resnet)
