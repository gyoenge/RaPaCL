# reference: scGPT 

from typing import List, Dict, Mapping, Optional, Tuple, Any, Union

import torch 
import torch.nn as nn 
from torch import Tensor 
from torch.autograd import Function

import logging
logger = logging.getLogger(__name__)


### About Batch Correction ### 

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


# The code is modified from https://github.com/wgchang/DSBN/blob/master/model/dsbn.py
class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(
        self,
        num_features: int,
        num_domains: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super(_DomainSpecificBatchNorm, self).__init__()
        self._cur_domain = None
        self.num_domains = num_domains
        self.bns = nn.ModuleList(
            [
                self.bn_handle(num_features, eps, momentum, affine, track_running_stats)
                for _ in range(num_domains)
            ]
        )

    @property
    def bn_handle(self) -> nn.Module:
        raise NotImplementedError

    @property
    def cur_domain(self) -> Optional[int]:
        return self._cur_domain

    @cur_domain.setter
    def cur_domain(self, domain_label: int):
        self._cur_domain = domain_label

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, domain_label: int) -> torch.Tensor:
        self._check_input_dim(x)
        if domain_label >= self.num_domains:
            raise ValueError(
                f"Domain label {domain_label} exceeds the number of domains {self.num_domains}"
            )
        bn = self.bns[domain_label]
        self.cur_domain = domain_label
        return bn(x)


class DomainSpecificBatchNorm1d(_DomainSpecificBatchNorm):
    @property
    def bn_handle(self) -> nn.Module:
        return nn.BatchNorm1d

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() > 3:
            raise ValueError(
                "expected at most 3D input (got {}D input)".format(input.dim())
            )


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    @property
    def bn_handle(self) -> nn.Module:
        return nn.BatchNorm2d

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

