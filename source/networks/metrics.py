# metrics.py
# © 2020 Giorgia Adorni and Elia Cereda
# Adapted from https://github.com/GiorgiaAuroraAdorni/learning-relative-interactions-through-imitation

import pandas as pd
from tqdm import tqdm


class StreamingMean:
    """
    Compute the (possibly weighted) mean of a sequence of values in streaming fashion.

    This class stores the current mean and current sum of the weights and updates
    them when a new data point comes in.

    This should have better stability than summing all samples and dividing at the
    end, since here the partial mean is always kept at the same scale as the samples.
    """
    def __init__(self):
        self.reset()

    def update(self, sample, weight=1.0):
        """

        :param sample: sample
        :param weight: weight
        """
        self._weights += weight
        self._mean += weight / self._weights * (sample - self._mean)

    def reset(self):
        """

        """
        self._weights = 0.0
        self._mean = 0.0

    @property
    def mean(self):
        """

        :return self._mean: mean
        """
        return self._mean


class NetMetrics:
    """
    This class is supposed to create a dataframe that collects, updates and saves to file the metrics of a model.

    :param t: tqdm
    :param metrics_path: file where to save the metrics
    """

    TRAIN_LOSS = 't. loss'
    VALIDATION_LOSS = 'v. loss'

    def __init__(self, t: tqdm, metrics_path):
        self.metrics_path = metrics_path
        self.df = pd.DataFrame(columns=[
            self.TRAIN_LOSS, self.VALIDATION_LOSS
        ])

        self.t = t

    def update(self, train_loss, valid_loss):
        """
        :param train_loss: training loss
        :param valid_loss: validation loss
        """
        metrics = {self.TRAIN_LOSS: float(train_loss), self.VALIDATION_LOSS: float(valid_loss)}
        self.df = self.df.append(metrics, ignore_index=True)

    def finalize(self):
        """

        :return:
        """
        self.df.to_pickle(self.metrics_path)
