"""Uncertainty objective retained from the supplied research code."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.pos_weight is not None and self.pos_weight.device != inputs.device:
            self.pos_weight = self.pos_weight.to(inputs.device)

        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            reduction='none',
            pos_weight=self.pos_weight
        )
        prob = torch.sigmoid(inputs)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        modulating_factor = (1 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = ce_loss * modulating_factor * alpha_factor

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class HeteroscedasticFocalLoss(nn.Module):
    """Monte-Carlo focal loss for a logistic-normal binary classifier.

    The model predicts a latent logit distribution N(mean, variance). Sampling
    through the reparameterization trick makes both the mean and variance heads
    trainable. A small variance penalty prevents an unbounded-variance shortcut.
    """

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None,
                 num_samples=8, variance_regularizer=1e-4):
        super().__init__()
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        self.num_samples = int(num_samples)
        self.variance_regularizer = float(variance_regularizer)
        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            pos_weight=pos_weight,
            reduction='mean'
        )

    def forward(self, mean, variance, targets):
        variance = variance.clamp(min=1e-6, max=25.0)
        std = torch.sqrt(variance)
        noise = torch.randn(
            (self.num_samples,) + tuple(mean.shape),
            device=mean.device,
            dtype=mean.dtype
        )
        sampled_logits = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        sampled_targets = targets.unsqueeze(0).expand_as(sampled_logits)
        predictive_loss = self.focal(sampled_logits, sampled_targets)
        return predictive_loss + self.variance_regularizer * variance.mean()


def logistic_normal_probability(mean, variance):
    """Deterministic approximation of E[sigmoid(z)] for z ~ N(mean, var)."""
    variance = variance.clamp(min=1e-6, max=25.0)
    scale = torch.sqrt(1.0 + (math.pi / 8.0) * variance)
    return torch.sigmoid(mean / scale)

