"""Loss and metrics. Pixels labelled IGNORE (the trimap boundary band) count for neither."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IGNORE


class CEDiceLoss(nn.Module):
    """Cross-entropy + soft Dice."""

    def __init__(self, weight=None, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        valid = (target != IGNORE).unsqueeze(1)
        prob = logits.float().softmax(dim=1) * valid
        onehot = F.one_hot(torch.where(target == IGNORE, 0, target), logits.shape[1]).permute(0, 3, 1, 2) * valid
        inter = (prob * onehot).sum(dim=(0, 2, 3))
        denom = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
        return ce + self.dice_weight * dice


def confusion(pred, target, n):
    k = target != IGNORE
    return torch.bincount(target[k] * n + pred[k], minlength=n * n).reshape(n, n)


def scores(cm, class_names):
    cm = cm.double()
    tp = cm.diag()
    iou = tp / (cm.sum(0) + cm.sum(1) - tp).clamp(min=1)
    dice = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp(min=1)
    return {"miou": iou.mean().item(), "pixel_acc": (tp.sum() / cm.sum().clamp(min=1)).item(),
            **{f"iou_{c}": iou[i].item() for i, c in enumerate(class_names)},
            **{f"dice_{c}": dice[i].item() for i, c in enumerate(class_names)}}


def pet_iou(pred, target):
    """Per-image pet-vs-background IoU over non-ignored pixels."""
    valid = target != IGNORE
    p, t = (pred != 0) & valid, (target != 0) & valid
    inter = (p & t).flatten(1).sum(1).float()
    union = (p | t).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
