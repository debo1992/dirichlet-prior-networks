

from torch import nn
import torch 
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

def one_hot(labels):
    # Credits: ptrblck
    # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/31

    x = torch.zeros(len(labels), self.num_labels)
    x.scatter_(1, labels.unsqueeze(1), 1.)
    return x

class NLLCost(nn.Module):
    def __init__(self, num_labels, eps=1e-8):
        super().__init__()
        self.eps = 1e-8
        self.num_labels = num_labels

    def forward(self, net_output, labels):
        # Translating below segment to PyTorch. This one is present in  the paper.
        # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/
        # 79cb8300238271566a5bbb69f0744f1d80924a1a/
        # prior_networks/dirichlet/dirichlet_prior_network.py
        # #L328-L334

        logits = net_output['logits']
        targets = one_hot(labels)

        B, H = logits.size()
        dimB, dimH = 0, 1
        alphas = logits.exp() + self.eps
        concentration =  alphas.sum(axis = dimH)
        loss = (
                torch.lgamma(concentration)
                - torch.lgamma(alphas).sum(dim = dimH) 
                + torch.sum(
                    (alphas - 1.0) * targets.float().log(),
                    dim = dimH
                )
        )

        # In the end, this looks like a complicated formulation of a
        # probability using dirichlet components.
        # logits can be trained with crossentropy, which should be
        # equivalent, why are we going through this pain?

        return loss

class ExpectedKL(nn.Module):
    def __init__(self, alpha, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, net_output, labels):
        # TODO(jerin): Make one more pass. There may be a more
        # numerically stable way of doing this.
        mean = net_output['mean']
        precision = net_output['precision']

        B, H = mean.size()
        dimB, dimH = 0, 1
        eps = self.eps

        target_mean = one_hot(labels)
        target_precision = self.alpha * torch.ones((B, 1)).float()

        loss = (
            torch.lgamma(target_precision + eps) 
            - torch.lgamma(precision + eps)
            + torch.sum(
                (
                    torch.lgamma(mean * precision + eps)
                    - torch.lgamma(target_mean * target_precision + eps)
                ), dim=dimH
            )
            + (
                torch.sum(
                    (target_precision * target_mean - precision * mean) 
                    * (
                        torch.digamma(target_mean * target_precision + eps) 
                        - torch.digamma(target_precision + eps)
                    ), dim = dimH
        )

        loss = loss.mean()
        return loss
