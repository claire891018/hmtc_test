#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from helper.utils import get_hierarchy_relations


class FocalBCEWithLogitsLoss(nn.Module):
    """
    Multi-label focal loss (BCEWithLogits 版本)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha  # None or float (pos class alpha)
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits/targets: [B, N]
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B,N]
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)  # p if y=1 else (1-p)

        modulating = (1.0 - p_t).pow(self.gamma)  # [B,N]

        if self.alpha is not None:
            # alpha for positive, (1-alpha) for negative
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * modulating * bce
        else:
            loss = modulating * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    ASL (Asymmetric Loss) for multi-label classification
    常用於 extreme multi-label：對 negative 做更強的 focusing
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction="mean"):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip) if clip is not None else 0.0
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits/targets: [B, N]
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # optional clipping for negatives (常見 trick)
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # basic CE terms
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        # asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt_pos = xs_pos * targets
            pt_neg = xs_neg * (1.0 - targets)
            pt = pt_pos + pt_neg

            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = (1.0 - pt).pow(gamma)
            loss = (loss_pos + loss_neg) * focal_weight
        else:
            loss = loss_pos + loss_neg

        loss = -loss  # make positive

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ClassificationLoss(nn.Module):
    """
    - classification loss: BCE / Focal / ASL
    - recursive regularization: same as original
    - label weighting/masking: handle labels missing in TRAIN
    """
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 recursive_penalty,
                 recursive_constraint=True,
                 loss_type="BCEWithLogitsLoss",
                 # focal
                 focal_gamma=2.0,
                 focal_alpha=None,
                 # asl
                 asl_gamma_pos=0.0,
                 asl_gamma_neg=4.0,
                 asl_clip=0.05,
                 # label weighting
                 label_weight=None,   # Tensor [N] or None
                 label_mask=None      # Tensor [N] in {0,1} or None
                 ):
        super().__init__()

        self.loss_type = str(loss_type)

        if self.loss_type.lower() in ["bce", "bcewithlogitsloss"]:
            self.base_loss = nn.BCEWithLogitsLoss(reduction="none")
        elif self.loss_type.lower() in ["focal", "focalbce", "focal_bce"]:
            self.base_loss = FocalBCEWithLogitsLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="none")
        elif self.loss_type.lower() in ["asl", "asymmetric", "asymmetricloss"]:
            self.base_loss = AsymmetricLoss(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy, label_map)
        self.recursive_penalty = float(recursive_penalty)
        self.recursive_constraint = bool(recursive_constraint)

        # label_weight: [N] float
        # label_mask:  [N] 0/1
        self.register_buffer("label_weight", label_weight if label_weight is not None else None)
        self.register_buffer("label_mask", label_mask if label_mask is not None else None)

    def _recursive_regularization(self, params, device):
        rec_reg = 0.0
        for i in range(len(params)):
            if i not in self.recursive_relation.keys():
                continue
            child_list = self.recursive_relation[i]
            if not child_list:
                continue
            child_list = torch.tensor(child_list).to(device)
            child_params = torch.index_select(params, 0, child_list)
            parent_params = torch.index_select(params, 0, torch.tensor(i).to(device))
            parent_params = parent_params.repeat(child_params.shape[0], 1)
            _diff = parent_params - child_params
            diff = _diff.view(_diff.shape[0], -1)
            rec_reg += 1.0 / 2 * torch.norm(diff, p=2) ** 2
        return rec_reg

    def _apply_label_weight_mask(self, loss_matrix):
        """
        loss_matrix: [B, N]  (reduction none)
        """
        if self.label_mask is not None:
            # [N] -> [1,N]
            loss_matrix = loss_matrix * self.label_mask.view(1, -1)

        if self.label_weight is not None:
            loss_matrix = loss_matrix * self.label_weight.view(1, -1)

        # normalize: avoid changing scale too much when many labels are masked
        denom = loss_matrix.numel()
        if self.label_mask is not None:
            denom = self.label_mask.sum().clamp(min=1.0) * loss_matrix.size(0)
        return loss_matrix.sum() / denom

    def forward(self, logits, targets, recursive_params):
        device = logits.device

        # base loss (reduction none)
        if isinstance(self.base_loss, (nn.BCEWithLogitsLoss,)):
            loss_matrix = self.base_loss(logits, targets)  # [B,N]
        else:
            # our custom losses use reduction="none" => return [B,N]
            loss_matrix = self.base_loss(logits, targets)

        loss = self._apply_label_weight_mask(loss_matrix)

        if self.recursive_constraint and recursive_params is not None:
            loss = loss + self.recursive_penalty * self._recursive_regularization(recursive_params, device)

        return loss
