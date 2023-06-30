import re
from typing import Optional, Sequence

import config
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim

from learning.loss import ComposedLoss, DeepLoss, SoftDiceLoss


def dice_torch(true: torch.Tensor, pred: torch.Tensor):
    intersection = torch.logical_and(true, pred)
    return 2 * intersection.sum() / (true.sum() + pred.sum())


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


class PolyLrCallback(pl.Callback):
    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        lr = poly_lr(trainer.current_epoch, trainer.max_epochs, pl_module.initial_learning_rate, exponent=0.9)
        for param_group in trainer.lightning_optimizers[0].param_groups:
            param_group['lr'] = lr
        trainer.lightning_optimizers[0].defaults['lr'] = lr
        return super().on_epoch_end(trainer, pl_module)


class SegmentationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, initial_learning_rate: float = 1e-2):
        """Lightning module for brain tumor segmentation.

        Args:
            model: model to be trained
            learning_rate: optimizer learning rate
        """
        super().__init__()
        self.model = model

        loss_weights = torch.Tensor([1, 1]).to(self.device)
        dice_loss = SoftDiceLoss(batch_dice=True, smooth=1e-5, ignore_index=config.ignore_value, return_log=False)
        ce_loss = nn.CrossEntropyLoss(ignore_index=config.ignore_value)
        base_loss = ComposedLoss(loss_weights, [dice_loss, ce_loss])

        if self.model.output_all_levels:
            scale_weights = torch.Tensor([1 / 2**i for i in range(self.model.depth - 1)]).to(self.device)
            scale_weights[-1] = 0.0
            scale_weights = scale_weights / scale_weights.sum()
            loss = DeepLoss(base_loss, scale_weights)
        else:
            loss = base_loss

        self.n_warmup_epochs = 5
        self.initial_learning_rate = initial_learning_rate
        self.loss = loss

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model.forward(input)

    def log_metrics(self, output: Optional[Sequence[torch.Tensor]], target: torch.Tensor, is_training: bool):
        if output is None:
            return

        log_str = '' if is_training else 'val_'
        predicted_classes = torch.argmax(output[0], dim=1)

        keep_mask = target != config.ignore_value
        true_classes = target[keep_mask]
        predicted_classes = predicted_classes[keep_mask]

        accuracy = torch.sum(predicted_classes == true_classes) / torch.numel(true_classes)
        self.log(f'{log_str}accuracy', accuracy.item(), prog_bar=False, on_step=False, on_epoch=True)

        dices = []
        for class_i in torch.unique(true_classes):
            if class_i != 0:
                class_dice = dice_torch(true_classes == class_i, predicted_classes == class_i)
                dices.append(class_dice.item())
                self.log(f'{log_str}dice_{class_i}', class_dice.item(), prog_bar=True, on_step=False, on_epoch=True)

            predicted_occurences = (predicted_classes == class_i).sum().to(torch.float)
            self.log(f'{log_str}pred_occ_{class_i}', predicted_occurences.item(),
                     prog_bar=False, on_step=False, on_epoch=True)

            true_occurences = (true_classes == class_i).sum().to(torch.float)
            self.log(f'{log_str}true_occ_{class_i}', true_occurences.item(),
                     prog_bar=False, on_step=False, on_epoch=True)

        non_nan_dices = np.array(dices)[np.logical_not(np.isnan(dices))]
        mean_dice = 0 if len(non_nan_dices) == 0 else np.mean(non_nan_dices)
        self.log(f'{log_str}mean_dice', mean_dice, prog_bar=False, on_step=False, on_epoch=True)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        input, target = batch

        background_mask = None
        if self.current_epoch < self.n_warmup_epochs:
            background_mask = target == 0
            ignore_background_fraction = (self.n_warmup_epochs - self.current_epoch) / self.n_warmup_epochs
            ignore_mask = torch.logical_and(torch.bernoulli(
                ignore_background_fraction * torch.ones_like(target)) > .5, background_mask)
            target[ignore_mask] = config.ignore_value

        output = self.model.forward(input)
        loss = self.loss(output, target)
        self.log('loss', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        if background_mask is not None:
            target = target * torch.logical_not(background_mask)
        self.log_metrics(output, target, is_training=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        input, target = batch

        output = self.model.forward(input)
        loss = self.loss(output, target)

        self.log('val_loss', loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        self.log_metrics(output, target, is_training=False)

        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.initial_learning_rate,
                              momentum=0.99, weight_decay=3e-05, nesterov=True)
        return {'optimizer': optimizer, 'monitor': 'val_mean_dice'}


class BratsSegmentationModule(SegmentationModule):
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        name, model_input, target = batch

        output = self.model.forward(model_input)
        loss = self.loss(output, target)

        self.log('val_loss', loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        self.log_metrics(output, target, is_training=False)

        predicted_classes = torch.argmax(output[0], dim=1).to('cpu').to(dtype=torch.uint8)
        target = target.to('cpu', dtype=torch.uint8)

        # Go through samples and save data to be used for calculating volume metrics (all real validation data)
        targets, predictions, names = [], [], []
        for n, p, t in zip(name, predicted_classes, target):
            match = re.search(r'Subject_(\d+)', n)
            if match is None:
                p, t = None, None
            targets.append(t)
            predictions.append(p)
            names.append(n)
        return {'loss': loss, 'names': names, 'predictions': predictions, 'targets': targets}

    def calculate_volume_metrics(self, outputs, extra_str: str):
        # build volumes for all subjects, note that slice order might not be correct
        prediction_volumes = dict()
        target_volumes = dict()
        for output in outputs:
            for name, prediction_slice, target_slice in zip(output['names'], output['predictions'], output['targets']):
                match = re.search(r'Subject_(\d+)', name)
                if match is not None:
                    subject_index = int(match[1])
                    if subject_index in prediction_volumes:
                        prediction_volumes[subject_index] = torch.cat((prediction_volumes[subject_index],
                                                                       prediction_slice.unsqueeze(dim=0)))
                        target_volumes[subject_index] = torch.cat((target_volumes[subject_index],
                                                                   target_slice.unsqueeze(dim=0)))
                    else:
                        prediction_volumes[subject_index] = prediction_slice.unsqueeze(dim=0)
                        target_volumes[subject_index] = target_slice.unsqueeze(dim=0)

        dices = [[], [], []]
        for subject_i in prediction_volumes.keys():
            for i, class_i in enumerate((1, 2, 3)):
                this_dice = dice_torch(target_volumes[subject_i] == class_i, prediction_volumes[subject_i] == class_i)
                dices[i].append(this_dice.item())

        mean_dices = []
        for i, class_i in enumerate((1, 2, 3)):
            mean_dices.append(np.nanmean(dices[i]))
            self.log(f'{extra_str}_macro_dice_{class_i}', mean_dices[i], prog_bar=False, on_step=False, on_epoch=True)
        self.log(f'{extra_str}_macro_dice_mean', np.nanmean(mean_dices), prog_bar=False, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        self.calculate_volume_metrics(outputs, 'val')
