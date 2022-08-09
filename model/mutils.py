import torch.nn as nn

def accumulate(model1: nn.Module, model2: nn.Module, decay: float = 0.999):
    """Supporting function for EMA.
    Args:
        model1: first model
        model2: second model
        decay: coefficient for ema
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for key in par1.keys():
        par1[key].data.mul_(decay).add_(par2[key].data, alpha=1 - decay)