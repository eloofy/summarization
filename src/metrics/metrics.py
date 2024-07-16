from base.base_metric import BaseMetric
from metrics.custom import RougeCustom, BERTCustom
from typing import Any


class ROUGEMetric(BaseMetric):
    """
    Base rouge metric
    """

    def __init__(self, rouge_type: str, **kwargs: Any) -> None:
        super().__init__(metric=RougeCustom(metrics=[rouge_type]), **kwargs)


class BERTMetric(BaseMetric):
    """
    Base bert metric
    """

    def __init__(self, **kwargs):
        super().__init__(metric=BERTCustom(**kwargs))
