from typing import List, Any
from torchmetrics import MetricCollection
from src.metrics.metrics import BERTMetric, ROUGEMetric


def get_metrics(rouge_types: List[str], bert: bool, **kwargs: Any) -> MetricCollection:
    """
    Class for getting metrics
    :param rouge_types: rouge type
    :param bert: whatever to use metric
    :param kwargs: adds param
    :return:
    """
    metrics = {}
    for rouge_type in rouge_types:
        metrics[rouge_type] = ROUGEMetric(rouge_type=rouge_type, **kwargs)

    if bert:
        metrics["bert"] = BERTMetric(**kwargs)

    return MetricCollection(metrics)
