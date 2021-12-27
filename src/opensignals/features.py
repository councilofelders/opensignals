import logging
from typing import List, Optional, Protocol, Dict, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureGenerator(Protocol):
    def generate_features(self, ticker_data: pd.DataFrame, feature_prefix: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        pass


class VarChange(FeatureGenerator):
    def __init__(self, num_days: int = 1, variable: str = 'adj_close'):
        super().__init__()
        self.num_days = num_days
        self.variable = variable

    def generate_features(self,
                          ticker_data: pd.DataFrame,
                          feature_prefix: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        logger.info(f'generating var change {self.num_days} '
                    f'for {self.variable}...')
        feature_prefix_name = f'{self.variable}_x{self.num_days}'
        if feature_prefix:
            feature_prefix_name = f'{feature_prefix}_{feature_prefix_name}'

        ticker_groups = ticker_data.groupby('bloomberg_ticker')
        ticker_data[feature_prefix_name] = \
            ticker_groups[self.variable].transform(
                lambda x: x.shift(self.num_days))

        ticker_data[f'{feature_prefix_name}_diff'] = \
            ticker_data[self.variable] / ticker_data[feature_prefix_name] - 1
        return ticker_data, []


def _parse_num_days(num_days: Union[int, List[int]]) -> List[int]:
    if isinstance(num_days, int):
        # +1 to be backwards compatible
        steps = list(range(num_days + 1))
    elif isinstance(num_days, list):
        steps = sorted(num_days)
        if steps[0] != 0:
            steps = [0] + steps
    else:
        raise ValueError("`num_days` should be of int or list type")
    return steps


class RSI(FeatureGenerator):
    """Relative Strength Index"""

    def __init__(self, num_days: int = 5, interval: int = 10, variable: str = 'adj_close'):
        super().__init__()
        self.steps = _parse_num_days(num_days)
        self.interval = interval
        self.variable = variable

    @staticmethod
    def relative_strength_index(prices: pd.Series, interval: int) -> pd.Series:
        '''
        Computes Relative Strength Index given a price series and lookback
        interval
        See more here https://www.investopedia.com/terms/r/rsi.asp
        '''
        delta = prices.diff()

        # copy deltas, set losses to 0, get rolling avg
        gains = delta.copy()
        gains[gains < 0] = 0
        avg_gain = gains.rolling(interval).mean()

        # copy deltas, set gains to 0, get rolling avg
        losses = delta.copy()
        losses[losses > 0] = 0
        avg_loss = losses.rolling(interval).mean().abs()

        # calculate relative strength and it's index
        rel_strength = avg_gain / avg_loss
        rsi: pd.Series = 100.0 - (100.0 / (1.0 + rel_strength))
        return rsi

    def get_feature_names(self, prefix_name: str) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
        # define column names of features, target, and prediction
        feat_quintile_lag = {step: f'{prefix_name}_quintile_lag_{step}'
                             for step in self.steps}
        feat_rsi_diff = {step: f'{prefix_name}_diff_{step}'
                         for step in self.steps[:-1]}
        feat_rsi_diff_abs = {step: f'{prefix_name}_abs_diff_{step}'
                             for step in self.steps[:-1]}
        return feat_quintile_lag, feat_rsi_diff, feat_rsi_diff_abs

    def generate_features(self,
                          ticker_data: pd.DataFrame,
                          feature_prefix: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        # add Relative Strength Index
        logger.info(f'generating RSI {self.interval} for {self.variable}...')

        feature_prefix_name = f'RSI_{self.interval}_{self.variable}'
        if feature_prefix:
            feature_prefix_name = f'{feature_prefix}_{feature_prefix_name}'

        ticker_groups = ticker_data.groupby('bloomberg_ticker')
        ticker_data[feature_prefix_name] = \
            ticker_groups[self.variable].transform(
                lambda x: self.relative_strength_index(x, self.interval)
        )

        # group by era (date)
        logger.debug('grouping by dates...')
        date_groups = ticker_data.groupby('date')

        # create quintile labels within each era, useful for learning
        # srelative ranking
        logger.debug('generating RSI quintiles...')
        col = f'{feature_prefix_name}_quintile'
        ticker_data[col] = date_groups[feature_prefix_name].transform(
            lambda group: pd.qcut(group, 5, labels=False, duplicates='drop')
        )
        ticker_data.dropna(inplace=True)
        ticker_data[col] = ticker_data[col].astype('int8')

        (
            feat_quintile_lag, feat_rsi_diff, feat_rsi_diff_abs
        ) = self.get_feature_names(feature_prefix_name)

        # create lagged features grouped by ticker
        logger.debug('grouping by ticker...')
        ticker_groups = ticker_data.groupby('bloomberg_ticker')

        # lag 0 is that day's value, lag 1 is yesterday's value, etc
        logger.debug('generating lagged RSI quintiles...')
        for day in self.steps:
            col = f'{feature_prefix_name}_quintile'
            ticker_data[feat_quintile_lag[day]] = ticker_groups[col].transform(
                lambda group: group.shift(day)
            ).astype('float16')

        # create difference of the lagged features and
        # absolute difference of the lagged features (change in RSI
        # quintile by day)
        logger.debug('generating lagged RSI diffs...')

        for i in range(len(self.steps) - 1):
            step = self.steps[i]
            ticker_data[feat_rsi_diff[step]] = (
                ticker_data[feat_quintile_lag[step]] -
                ticker_data[feat_quintile_lag[self.steps[i + 1]]]
            )
            ticker_data[feat_rsi_diff_abs[step]] = \
                np.abs(ticker_data[feat_rsi_diff[step]])

        feature_names = (list(feat_quintile_lag.values()) +
                         list(feat_rsi_diff.values()) +
                         list(feat_rsi_diff_abs.values()))
        return ticker_data, feature_names


class SMA(FeatureGenerator):
    """Simple Moving Average"""

    def __init__(self, num_days: int = 5, interval: int = 10, variable: str = 'adj_close'):
        super().__init__()
        self.steps = _parse_num_days(num_days)
        self.interval = interval
        self.variable = variable

    @staticmethod
    def simple_moving_average(prices: pd.Series, interval: int) -> pd.Series:
        return prices.rolling(interval).mean()   # type: ignore

    def get_feature_names(self, prefix_name: str) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
        # define column names of features, target, and prediction
        feat_quintile_lag = {step: f'{prefix_name}_quintile_lag_{step}'
                             for step in self.steps}
        feat_rsi_diff = {step: f'{prefix_name}_diff_{step}'
                         for step in self.steps[:-1]}
        feat_rsi_diff_abs = {step: f'{prefix_name}_abs_diff_{step}'
                             for step in self.steps[:-1]}
        return feat_quintile_lag, feat_rsi_diff, feat_rsi_diff_abs

    def generate_features(self,
                          ticker_data: pd.DataFrame,
                          feature_prefix: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        # add Relative Strength Index
        logger.info(f'generating SMA {self.interval} for {self.variable}...')

        feature_prefix_name = f'SMA_{self.interval}_{self.variable}'
        if feature_prefix:
            feature_prefix_name = f'{feature_prefix}_{feature_prefix_name}'

        ticker_groups = ticker_data.groupby('bloomberg_ticker')
        ticker_data[feature_prefix_name] = \
            ticker_groups[self.variable].transform(
                lambda x: self.simple_moving_average(x, self.interval)
        )

        # group by era (date)
        logger.debug('grouping by dates...')
        date_groups = ticker_data.groupby('date')

        # create quintile labels within each era, useful for
        # learning relative ranking
        logger.debug('generating SMA quintiles...')
        col = f'{feature_prefix_name}_quintile'
        ticker_data[col] = date_groups[feature_prefix_name].transform(
            lambda group: pd.qcut(group, 5, labels=False, duplicates='drop')
        )
        ticker_data.dropna(inplace=True)
        ticker_data[col] = ticker_data[col].astype('int8')

        (
            feat_quintile_lag, feat_sma_diff, feat_sma_diff_abs
        ) = self.get_feature_names(feature_prefix_name)

        # create lagged features grouped by ticker
        logger.debug('grouping by ticker...')
        ticker_groups = ticker_data.groupby('bloomberg_ticker')

        # lag 0 is that day's value, lag 1 is yesterday's value, etc
        logger.debug('generating lagged SMA quintiles...')
        for day in self.steps:
            col = f'{feature_prefix_name}_quintile'
            ticker_data[feat_quintile_lag[day]] = ticker_groups[col].transform(
                lambda group: group.shift(day)
            ).astype('float16')

        # create difference of the lagged features and
        # absolute difference of the lagged features (change in
        # SMA quintile by day)
        logger.debug('generating lagged SMA diffs...')
        for i in range(len(self.steps) - 1):
            ticker_data[feat_sma_diff[self.steps[i]]] = (
                ticker_data[feat_quintile_lag[self.steps[i]]] -
                ticker_data[feat_quintile_lag[self.steps[i + 1]]])
            ticker_data[feat_sma_diff_abs[self.steps[i]]] = \
                np.abs(ticker_data[feat_sma_diff[self.steps[i]]])

        feature_names = (list(feat_quintile_lag.values()) +
                         list(feat_sma_diff.values()) +
                         list(feat_sma_diff_abs.values()))
        return ticker_data, feature_names
