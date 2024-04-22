"""Implementation of AnomalyScoreThreshold based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import torch
from torch import Tensor
from torchmetrics import PrecisionRecallCurve


class AnomalyScoreThreshold(PrecisionRecallCurve):
    """Anomaly Score Threshold.

    This class computes/stores the threshold that determines the anomalous label
    given anomaly scores. If the threshold method is ``manual``, the class only
    stores the manual threshold values.

    If the threshold method is ``adaptive``, the class initially computes the
    adaptive threshold to find the optimal f1_score and stores the computed
    adaptive threshold value.
    """

    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
        super().__init__(num_classes=1, **kwargs)

        self.add_state("value", default=torch.tensor(default_value), persistent=True)  # pylint: disable=not-callable
        self.value = torch.tensor(default_value)  # pylint: disable=not-callable

    def compute(self) -> Tensor:
        """Compute the threshold that yields the optimal F1 score.

        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        current_targets = torch.concat(self.target)
        current_preds = torch.concat(self.preds)

        epsilon = 1e-3

        if len(current_targets.unique()) == 1:
            if current_targets.max() == 0:
                self.value = current_preds.max() + epsilon
            else:
                self.value = current_preds.min() - epsilon
        else:
            precision, recall, thresholds = super().compute()
            f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
            optimal_f1_score = torch.max(f1_score)

            if thresholds.nelement() == 1:
                # Particular case when f1 score is 1 and the threshold is unique
                self.value = thresholds
            else:
                if optimal_f1_score == 1:
                    # If there is a good boundary between good and bads we pick the average of the highest good
                    # and lowest bad
                    max_good_score = current_preds[torch.where(current_targets == 0)].max()
                    min_bad_score = current_preds[torch.where(current_targets == 1)].min()
                    self.value = (max_good_score + min_bad_score) / 2
                else:
                    self.value = thresholds[torch.argmax(f1_score)]

        return self.value
