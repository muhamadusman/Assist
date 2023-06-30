from typing import Optional, Sequence
from torch import nn
import torch
import torch.nn.functional as F


def torch_soft_tp_fp_fn_tn(net_output, gt, axes, keep_mask=None):
    """Compute soft tp, fp, fn, tn  from network output and gt

    Args:
        net_output: tensor with dimensions (B, C, W, H)
        gt: tensor of shape (B, W, H), (B, 1, W, H) or one hot encoded (B, C, W, H)
        axes: axes to reduce during summation
        keep_mask: keep mask of shape (B, W, H) (1 for use, 0 for ignore)

    Returns:
        tp, fp, fn, tn
    """

    with torch.no_grad():
        if len(net_output.shape) != len(gt.shape):
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            gt = torch.minimum(gt, net_output.shape[1]-1 * torch.ones_like(gt))
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if keep_mask is not None:
        tp = torch.stack(tuple(x_i * keep_mask for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * keep_mask for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * keep_mask for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * keep_mask for x_i in torch.unbind(tn, dim=1)), dim=1)

    if len(axes) > 0:
        tp = tp.sum(dim=axes)
        fp = fp.sum(dim=axes)
        fn = fn.sum(dim=axes)
        tn = tn.sum(dim=axes)

    return tp, fp, fn, tn


class DeepLoss(nn.Module):
    def __init__(self, loss_fct: nn.Module, weights: torch.Tensor):
        super().__init__()
        self.loss_fct = loss_fct
        self.weights = weights

    def forward(self, x, y):
        total_loss = self.weights[0] * self.loss_fct(x[0], y)
        for i in range(1, len(x)):
            if self.weights[i] != 0:
                total_loss += self.weights[i] * self.loss_fct(x[i], y)
        return total_loss


class ComposedLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, loss_fcts: Sequence[nn.Module]):
        super().__init__()
        assert len(weights) == len(loss_fcts)
        self.weights = weights
        self.loss_fcts = loss_fcts

    def forward(self, x, y):
        total_loss = self.weights[0] * self.loss_fcts[0](x, y)
        for i in range(1, len(self.weights)):
            if self.weights[i] != 0:
                total_loss += self.weights[i] * self.loss_fcts[i](x, y)
        return total_loss


class SoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=False, smooth: float = 1e-05, ignore_index=Optional[int], return_log: bool = False):
        """Soft Dice Loss

        Args:
            batch_dice: Calculate dice over entire batch if true (else separate for each sample)
            smooth: smooth factor
            ignore_index
            return_log
        """
        super().__init__()

        self.batch_dice = batch_dice
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.return_log = return_log

    def forward(self, x, y):
        if self.batch_dice:
            axes = [0] + list(range(2, x.ndim))
        else:
            axes = list(range(2, x.ndim))

        x_soft = F.softmax(x, dim=1)

        loss_mask = y != self.ignore_index
        tp, fp, fn, _ = torch_soft_tp_fp_fn_tn(x_soft, y, axes, loss_mask)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)
        dc = dc.mean()
        if self.return_log:
            dc = torch.log(dc)
        return -dc
