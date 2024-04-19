# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Implementation of Optimal F1 score based on TorchMetrics."""
from typing import Optional
import warnings

import torch
from torch import Tensor
from torchmetrics import Metric, PrecisionRecallCurve


class OptimalF1(Metric):
    """Optimal F1 Metric.

    Compute the optimal F1 score at the adaptive threshold, based on the F1 metric of the true labels and the
    predicted anomaly scores.
    """

    full_state_update: bool = False

    def __init__(self, num_classes: int, **kwargs) -> None:
        warnings.warn(
            DeprecationWarning(
                "OptimalF1 metric is deprecated and will be removed in a future release. The optimal F1 score for "
                "Anomalib predictions can be obtained by computing the adaptive threshold with the "
                "AnomalyScoreThreshold metric and setting the computed threshold value in TorchMetrics F1Score metric."
            )
        )
        super().__init__(**kwargs)
        self.precision_recall_curve = PrecisionRecallCurve(num_classes=num_classes, compute_on_step=False)
        self.threshold: torch.Tensor = torch.tensor(-1.0)

    def update(self, preds: Tensor, target: Tensor, *args, **kwargs) -> None:
        """Update the precision-recall curve metric."""
        del args, kwargs  # These variables are not used.

        self.precision_recall_curve.update(preds, target)

    def compute(self) -> Tensor:
        """Compute the value of the optimal F1 score.

        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor
        current_targets = torch.concat(self.precision_recall_curve.target)
        current_preds = torch.concat(self.precision_recall_curve.preds)

        epsilon = 1e-3
        if len(current_targets.unique()) == 1:
            optimal_f1_score = torch.tensor(1.0)

            if current_targets.max() == 0:
                self.threshold = current_preds.max() + epsilon
            else:
                self.threshold = current_preds.min() - epsilon

            return optimal_f1_score
        else:
            precision, recall, thresholds = self.precision_recall_curve.compute()
            f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
            optimal_f1_score = torch.max(f1_score)

            if thresholds.nelement() == 1:
                # Particular case when f1 score is 1 and the threshold is unique
                self.threshold = thresholds
            else:
                if optimal_f1_score == 1:
                    # If there is a good boundary between good and bads we pick the average of the highest good
                    # and lowest bad
                    max_good_score = current_preds[torch.where(current_targets == 0)].max()
                    min_bad_score = current_preds[torch.where(current_targets == 1)].min()
                    self.threshold = (max_good_score + min_bad_score) / 2
                else:
                    self.threshold = thresholds[torch.argmax(f1_score)]

            return optimal_f1_score

    def reset(self) -> None:
        """Reset the metric."""
        self.precision_recall_curve.reset()
