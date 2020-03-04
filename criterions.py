from torch import nn
import torch 
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from collections import namedtuple
from dpn.constants import EPS, DatasetType
from ast import literal_eval


def one_hot(labels, num_labels):
    # Credits: ptrblck
    # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/31

    x = labels.new_zeros((len(labels), num_labels))
    x.scatter_(1, labels.unsqueeze(1), 1.)
    return x

def label_smooth(labels_one_hot, smoothing):
    if smoothing < EPS:
        return labels_one_hot

    batch_size, num_classes = labels_one_hot.size()
    smoothed = (
            (1 - num_classes*smoothing)*labels_one_hot 
            + smoothing * torch.ones_like(labels_one_hot)
    )
    return smoothed

def dirichlet_params_from_logits(logits):
    dimB, dimH = 0, 1
    alphas = logits.exp() + EPS
    alpha_0 = alphas.sum(dim = dimH, keepdim=True)
    return alphas, alpha_0


def lgamma(tensor):
    # Some confusion with lgamma's missing documentation.
    return torch.lgamma(tensor)


class Cost(nn.Module):
    @classmethod
    def build(cls, args):
        return cls()
    
class MultiTaskLoss(nn.Module):
    def __init__(self, weighted_losses):
        super().__init__()
        self.losses = weighted_losses

    def forward(self, net_output, labels, **kwargs):
        accumulator = 0
        for loss in self.losses:
            if loss.weight:
                accumulator += loss.weight * loss.f(net_output, labels, **kwargs)
        return accumulator



class DirichletKLDiv(Cost):
    def __init__(self, alpha, reduce=True, smoothing=1e-2):
        super().__init__()
        self.alpha = alpha
        self.reduce = reduce
        self.smoothing = smoothing

    def forward(self, net_output, labels, in_domain=True, **kwargs):
        # Translation of
        # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network.py#L281-L294

        logits, gain = net_output['logits'], net_output['gain']

        batch_size, num_classes = logits.size()
        dimB, dimH = 0, 1

        # mean and precision from the network
        mean = F.softmax(logits, dim=dimH)
        precision = torch.sum((logits + gain).exp(), dim=dimH, keepdim=True)


        def in_domain_targets():
            # the expected mean and precision, from the ground truth
            labels_one_hot = one_hot(labels, num_classes).float()
            target_mean = label_smooth(labels_one_hot, self.smoothing)
            target_precision = self.alpha * precision.new_ones((batch_size, 1))
            '''target means \mu = smoothed labels and target precision \alpha_0 = 1000'''
            return target_mean, target_precision

        def out_of_domain_targets():
            # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network.py#L619-L623
            target_precision = num_classes * precision.new_ones((batch_size, 1)).float()
            target_mean = torch.ones_like(mean).float()/num_classes
            return target_mean, target_precision

        target_f = in_domain_targets if in_domain else out_of_domain_targets
        target_mean, target_precision = target_f()

        loss = self._compute_loss(mean, precision, target_mean, target_precision)
        return loss

    def _compute_loss(self, mean, precision, target_mean, target_precision):
        # lgamma = log(gamma())
#        print(precision)
        eps = EPS
        dimB, dimH = 0, 1
        dlgamma = lgamma(target_precision + eps) - lgamma(precision + eps)
        dsumlgamma = torch.sum(
            (lgamma(mean * precision + eps)
              - lgamma(target_mean * target_precision + eps)
            ), dim = dimH
        )

        dconc = target_precision * target_mean - precision * mean
        dphiconc = (
            torch.digamma(target_mean * target_precision + eps) 
            - torch.digamma(target_precision + eps)
        )

        dprodconc_conc_phi = torch.sum(dconc*dphiconc, dim = dimH)
#        dprodconc_conc_phi = 
        loss = (dlgamma.squeeze() + dsumlgamma + dprodconc_conc_phi)
        loss = loss.mean()
        return loss

    @classmethod
    def build(cls, args):
        return cls(args.alpha)

def build_criterion(args):
    # Switch to control criterion
    # Criterion is a multi-task-objective.
    # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network.py#L629-L640

    loss_functions = {
        "dirichlet_kldiv": DirichletKLDiv,
    }  

    def validate_keys(weights): 
        wkeys = set(list(weights.keys()))
        lkeys = set(list(loss_functions.keys()))
        assert (wkeys <= lkeys), "Check loss supplied"

    def build_weighted_loss(weights):
        WeightedLoss = namedtuple('WeightedLoss', 'weight f')
        weighted_losses = [
            WeightedLoss(weight=weights[fname], f=loss_functions[fname].build(args))
            for fname in set(list(weights.keys()))
#            for fname in weights.keys()
        ]
        criterion = MultiTaskLoss(weighted_losses)
#        criterion = DirichletKLDiv(weighted_losses)
        return criterion

    ind_weights = literal_eval(args.ind_loss)
    ood_weights = literal_eval(args.ood_loss)
#    ind_weights = { "dirichlet_kldiv": DirichletKLDiv}
#    ood_weights = { "dirichlet_kldiv": DirichletKLDiv}

    validate_keys(ind_weights)
    validate_keys(ood_weights)

    criterion = {
        DatasetType.InD: build_weighted_loss(ind_weights),
        DatasetType.OoD: build_weighted_loss(ood_weights)
        
    }

    return criterion
