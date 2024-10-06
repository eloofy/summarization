from rouge import Rouge
from bert_score import score


class RougeCustom(Rouge):
    """
    Rouge metric
    """
    def __call__(self, hyp, ref):
        return [
            metric_score[self.metrics[0]]["f"]
            for metric_score in self.get_scores(hyp, ref, ignore_empty=True)
        ]


class BERTCustom:
    """
    Bert bert metric
    """

    def __init__(self, lang: str, verbose: bool, **kwargs):
        self.lang = lang
        self.verbose = verbose

    def __call__(self, hyp, ref, **kwargs):
        return score(hyp, ref, lang=self.lang, verbose=self.verbose)
