from torchmetrics import Metric
from abc import ABC
from typing import Any, List
import torch
import functools


def set_current_device(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.to(torch.cuda.current_device())
        self.results_metric = self.results_metric.to(torch.cuda.current_device())
        return func(self, *args, **kwargs)

    return wrapper


class BaseMetric(Metric, ABC):
    """
    Class base metric

    Load, save and compute
    """

    def __init__(self, metric: Any, **kwargs: Any) -> None:
        """

        :param metric: metric to compute
        :param kwargs: adds params
        """
        dist_sync_on_step = kwargs.pop("dist_sync_on_step", False)
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.metric = metric
        self.results_metric: torch.Tensor = torch.tensor([])

    @set_current_device
    def update(self, hypothesis: List[str], reference: List[str]) -> None:
        """
        Update metric
        :param hypothesis: predicted summ
        :param reference:  real summ
        """
        scores = self.metric(hypothesis, reference)
        if all(isinstance(item_sc, torch.Tensor) for item_sc in scores):
            scores = tuple(x.to(self.device) for x in scores)
            self.results_metric = torch.cat((self.results_metric, *scores))
        else:
            self.results_metric = torch.cat(
                (self.results_metric, torch.tensor(scores, device=self.device)),
            )

    def compute(self):
        """
        Compute mean metric
        """
        return self.results_metric.mean()

    def reset(self) -> None:
        """
        Update metric after epoch
        :return:
        """
        self.results_metric = torch.tensor([]).to(torch.cuda.current_device())