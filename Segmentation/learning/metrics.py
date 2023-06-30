from typing import Optional, Sequence

import numpy as np
from monai.metrics.utils import get_mask_edges
from scipy.ndimage import distance_transform_edt


def distance_metrics(pred_mask, gt_mask):
    mm_per_pixel = np.array([1.0, 1.0, 1.0])
    edges_pred, edges_gt = get_mask_edges(pred_mask, gt_mask)

    max_surface_distances = []
    average_surface_distances = []
    for a, b in zip((edges_pred, edges_gt), (edges_gt, edges_pred)):
        distance_map = distance_transform_edt(~b, sampling=mm_per_pixel)
        surface_distances = distance_map[a]
        average_surface_distances.append(np.nan if surface_distances.shape == (0,) else surface_distances.mean())
        max_surface_distances.append(np.nan if surface_distances.shape == (0,) else surface_distances.max())

    average_surface_distance = np.mean(average_surface_distances)
    hausdorff_distance = np.max(max_surface_distances)
    return hausdorff_distance, average_surface_distance


def add_metrics_from_mask(results, true_mask, pred_mask, class_key):
    tp = np.logical_and(true_mask, pred_mask).sum()
    fp = np.logical_and(true_mask, np.logical_not(pred_mask)).sum()
    fn = np.logical_and(np.logical_not(true_mask), pred_mask).sum()
    true_count = true_mask.sum()
    pred_count = pred_mask.sum()

    dice = 2 * tp / (2 * tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    results['dice'][class_key] = dice
    results['precision'][class_key] = precision
    results['recall'][class_key] = recall
    results['true_count'][class_key] = true_count
    results['pred_count'][class_key] = pred_count
    results['tp'][class_key] = tp
    results['fp'][class_key] = fp
    results['fn'][class_key] = fn

    hausdorff_distance, average_surface_distance = distance_metrics(pred_mask, true_mask)
    results['hausdorff_distance'][class_key] = hausdorff_distance
    results['average_surface_distance'][class_key] = average_surface_distance


def pixelwise_evaluation_metrics(true: np.ndarray,
                                 pred: np.ndarray,
                                 classes: Optional[Sequence[int]] = None,
                                 include_foreground_metrics: bool = True):
    if classes is None:
        classes = np.unique(true)

    metric_names = ('dice', 'precision', 'recall', 'true_count', 'pred_count', 'tp',
                    'fp', 'fn', 'hausdorff_distance', 'average_surface_distance')
    results = {metric_name: dict() for metric_name in metric_names}

    for cl in classes:
        if cl == 0 or cl == 255:
            continue
        true_mask = true == cl
        pred_mask = pred == cl

        add_metrics_from_mask(results, true_mask, pred_mask, cl)

    if include_foreground_metrics:
        for metric_name in metric_names:
            mean_of_metric = np.nanmean(list(results[metric_name].values()))
            results[metric_name][-1] = mean_of_metric

        add_metrics_from_mask(results, true > 0, pred > 0, 0)  # foreground metrics

    if include_foreground_metrics:
        add_metrics_from_mask(results, true > 0, pred > 0, 0)  # foreground metrics

    return results
