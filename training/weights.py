import torch.nn as nn


def initialize_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__

    if "Conv" in classname:
        nn.init.normal_(
            module.weight.data,
            mean=0.0,
            std=0.02,
        )

    elif "BatchNorm" in classname:
        nn.init.normal_(
            module.weight.data,
            mean=1.0,
            std=0.02,
        )

        nn.init.constant_(
            module.bias.data,
            0.0,
        )