
import torch.nn.functional as F

def dpo_loss(
    logp_chosen,
    logp_rejected,
    beta=0.1
):
    """
    logp_*: [batch]
    """
    diff = logp_chosen - logp_rejected
    return -F.logsigmoid(beta * diff).mean()
    # return F.softplus(-beta * diff).mean()
    # return (F.softplus(-beta * diff) + beta * diff).mean()
    # return (F.softplus(-beta * diff) * 2).mean()
    # return (F.softplus(-beta * diff) + F.softplus(beta * diff)).mean()
    # return (F.softplus(-beta * diff) * 2 + beta * diff).mean()
    # return (F.softplus(-beta * diff) * 2 - beta * diff).mean()
    # return (F.softplus(-beta * diff) * 4).mean()
    # return (F.softplus(-beta * diff) * 4 + beta * diff).mean()