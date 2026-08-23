import torch.nn as nn


def get_adversarial_loss() -> nn.Module:
    return nn.BCEWithLogitsLoss()