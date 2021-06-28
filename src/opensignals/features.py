import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VarChange:
    def __init__(self, num_days=1, variable='adj_close'):
        self.num_days=num_days
        self.variable=variable

    def generate_features(self, ticker_data):
        logger.info(f'generating var change {self.num_days} for {self.variable}...')
        feature_prefix_name = f'{self.variable}_x{self.num_days}'
        ticker_groups = ticker_data.groupby('bloomberg_ticker')
        ticker_data[feature_prefix_name] = ticker_groups[self.variable].transform(
            lambda x: x.shift(self.num_days)
        )

        ticker_data[f'{feature_prefix_name}_diff'] = ticker_data[self.variable] / ticker_data[feature_prefix_name] - 1
        return ticker_data, []


class RSI:
    def __init__(self, num_days=5, interval=10, variable='adj_close'):
        self.num_days=num_days
        self.interval=interval
        self.variable=variable

    @staticmethod
    def relative_strength_index(prices, interval):
        '''
        Computes Relative Strength Index given a price series and lookback interval
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
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def get_feature_names(num_days, prefix_name):
        # define column names of features, target, and prediction
        feat_quintile_lag = [f'{prefix_name}_quintile_lag_{num}' for num in range(num_days + 1)]
        feat_rsi_diff = [f'{prefix_name}_diff_{num}' for num in range(num_days)]
        feat_rsi_diff_abs = [f'{prefix_name}_abs_diff_{num}' for num in range(num_days)]
        return feat_quintile_lag, feat_rsi_diff, feat_rsi_diff_abs

    def generate_features(self, ticker_data):
        # add Relative Strength Index
        logger.info(f'generating RSI {self.interval} for {self.variable}...')
        feature_prefix_name = f'RSI_{self.interval}_{self.variable}'
        ticker_groups = ticker_data.groupby('bloomberg_ticker')
        ticker_data[feature_prefix_name] = ticker_groups[self.variable].transform(
            lambda x: self.relative_strength_index(x, self.interval)
        )

        # group by era (date)
        logger.debug('grouping by dates...')
        date_groups = ticker_data.groupby('date')

        # create quintile labels within each era, useful for learning relative ranking
        logger.debug('generating RSI quintiles...')
        ticker_data[f'{feature_prefix_name}_quintile'] = date_groups[feature_prefix_name].transform(
            lambda group: pd.qcut(group, 5, labels=False, duplicates='drop')
        )
        ticker_data.dropna(inplace=True)

        (
            feat_quintile_lag, feat_rsi_diff, feat_rsi_diff_abs
        ) = self.get_feature_names(self.num_days, feature_prefix_name)

        # create lagged features grouped by ticker
        logger.debug('grouping by ticker...')
        ticker_groups = ticker_data.groupby('bloomberg_ticker')

        # lag 0 is that day's value, lag 1 is yesterday's value, etc
        logger.debug('generating lagged RSI quintiles...')
        for day in range(self.num_days + 1):
            ticker_data[feat_quintile_lag[day]] = ticker_groups[f'{feature_prefix_name}_quintile'].transform(
                lambda group: group.shift(day)
            )

        # create difference of the lagged features and
        # absolute difference of the lagged features (change in RSI quintile by day)
        logger.debug('generating lagged RSI diffs...')
        for day in range(self.num_days):
            ticker_data[feat_rsi_diff[day]] = (
                ticker_data[feat_quintile_lag[day]] - ticker_data[feat_quintile_lag[day + 1]]
            )
            ticker_data[feat_rsi_diff_abs[day]] = np.abs(ticker_data[feat_rsi_diff[day]])

        return ticker_data, feat_quintile_lag + feat_rsi_diff + feat_rsi_diff_abs


class SMA:
    def __init__(self,  num_days=5, interval=10, variable='adj_close'):
        self.num_days=num_days
        self.interval=interval
        self.variable=variable

    @staticmethod
    def simple_moving_average(prices, interval):
        return prices.rolling(interval).mean()

    @staticmethod
    def get_feature_names(num_days, prefix_name):
        # define column names of features, target, and prediction
        feat_quintile_lag = [f'{prefix_name}_quintile_lag_{num}' for num in range(num_days + 1)]
        feat_rsi_diff = [f'{prefix_name}_diff_{num}' for num in range(num_days)]
        feat_rsi_diff_abs = [f'{prefix_name}_abs_diff_{num}' for num in range(num_days)]
        return feat_quintile_lag, feat_rsi_diff, feat_rsi_diff_abs

    def generate_features(self, ticker_data):
        # add Relative Strength Index
        logger.info(f'generating SMA {self.interval} for {self.variable}...')
        feature_prefix_name = f'SMA_{self.interval}_{self.variable}'
        ticker_groups = ticker_data.groupby('bloomberg_ticker')
        ticker_data[feature_prefix_name] = ticker_groups[self.variable].transform(
            lambda x: self.simple_moving_average(x, self.interval)
        )

        # group by era (date)
        logger.debug('grouping by dates...')
        date_groups = ticker_data.groupby('date')

        # create quintile labels within each era, useful for learning relative ranking
        logger.debug('generating SMA quintiles...')
        ticker_data[f'{feature_prefix_name}_quintile'] = date_groups[feature_prefix_name].transform(
            lambda group: pd.qcut(group, 5, labels=False, duplicates='drop')
        )
        ticker_data.dropna(inplace=True)

        (
            feat_quintile_lag, feat_sma_diff, feat_sma_diff_abs
        ) = self.get_feature_names(self.num_days, feature_prefix_name)

        # create lagged features grouped by ticker
        logger.debug('grouping by ticker...')
        ticker_groups = ticker_data.groupby('bloomberg_ticker')

        # lag 0 is that day's value, lag 1 is yesterday's value, etc
        logger.debug('generating lagged SMA quintiles...')
        for day in range(self.num_days + 1):
            ticker_data[feat_quintile_lag[day]] = ticker_groups[f'{feature_prefix_name}_quintile'].transform(
                lambda group: group.shift(day)
            )

        # create difference of the lagged features and
        # absolute difference of the lagged features (change in SMA quintile by day)
        logger.debug('generating lagged SMA diffs...')
        for day in range(self.num_days):
            ticker_data[feat_sma_diff[day]] = (
                ticker_data[feat_quintile_lag[day]] - ticker_data[feat_quintile_lag[day + 1]]
            )
            ticker_data[feat_sma_diff_abs[day]] = np.abs(ticker_data[feat_sma_diff[day]])

        return ticker_data, feat_quintile_lag + feat_sma_diff + feat_sma_diff_abs
