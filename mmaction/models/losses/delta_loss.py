import torch.nn.functional as F
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class DeltaLoss:
    def __init__(self, alpha=0.9, delta_nllloss=False, scale=False):
        self.alpha = alpha
        self.nllloss = nn.NLLLoss()
        self.delta_nllloss = delta_nllloss
        self.nllloss_delta = nn.NLLLoss(reduction="sum")
        self.scale = scale

    def _deltaloss(self, probs, target, correct):
        total = (~correct).sum()
        scores = F.softmax(probs, dim=1)
        probs = F.log_softmax(probs, dim=1)
        if self.delta_nllloss:
            lost = self.nllloss_delta(probs[correct], target[correct])
            found = total - self.nllloss_delta(probs[~correct], target[~correct])
        else:
            # weights = self.class_weights.gather(0, target).unsqueeze(1)
            # total = (weights[~correct]).sum()
            scores = scores.gather(1, target.unsqueeze(1))
            lost = (1 - scores[correct]).sum()
            found = (scores[~correct]).sum()
        if total > 0:
            if self.scale:
                deltaloss = (found - lost - total) / (-2 * total)
            else:
                deltaloss = (found - lost - total) / (-1 * total)
        else:
            deltaloss = lost
        return deltaloss

    def __call__(self, probs, target):
        nllloss = self.nllloss(F.log_softmax(probs, dim=1), target)
        correct = target == 8.0
        deltaloss = self._deltaloss(probs, target, correct)
        loss = (1 - self.alpha) * nllloss + self.alpha * deltaloss
        return loss