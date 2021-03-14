import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RSI:
    def __init__(self, num_days=5, interval=10):
        self.num_days=num_days
        self.interval=interval

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
    def get_rsi_feature_names(num_days):
        # define column names of features, target, and prediction
        feat_quintile_lag = [f'RSI_quintile_lag_{num}' for num in range(num_days + 1)]
        feat_rsi_diff = [f'RSI_diff_{num}' for num in range(num_days)]
        feat_rsi_diff_abs = [f'RSI_abs_diff_{num}' for num in range(num_days)]
        return feat_quintile_lag, feat_rsi_diff, feat_rsi_diff_abs

    def generate_features(self, ticker_data):
        # add Relative Strength Index
        logger.info('generating RSI for each price...')
        ticker_groups = ticker_data.groupby('ticker')
        ticker_data['RSI'] = ticker_groups['adj_close'].transform(
            lambda x: self.relative_strength_index(x, self.interval)
        )

        # group by era (date)
        logger.info('grouping by dates...')
        date_groups = ticker_data.groupby('date')

        # create quintile labels within each era, useful for learning relative ranking
        logger.info('generating RSI quintiles...')
        ticker_data['RSI_quintile'] = date_groups['RSI'].transform(
            lambda group: pd.qcut(group, 5, labels=False, duplicates='drop')
        )
        ticker_data.dropna(inplace=True)

        feat_quintile_lag, feat_rsi_diff, feat_rsi_diff_abs = self.get_rsi_feature_names(self.num_days)

        # create lagged features grouped by ticker
        logger.info('grouping by ticker...')
        ticker_groups = ticker_data.groupby('ticker')

        # lag 0 is that day's value, lag 1 is yesterday's value, etc
        logger.info('generating lagged RSI quintiles...')
        for day in range(self.num_days + 1):
            ticker_data[feat_quintile_lag[day]] = ticker_groups['RSI_quintile'].transform(
                lambda group: group.shift(day)
            )

        # create difference of the lagged features and
        # absolute difference of the lagged features (change in RSI quintile by day)
        logger.info('generating lagged RSI diffs...')
        for day in range(self.num_days):
            ticker_data[feat_rsi_diff[day]] = (
                ticker_data[feat_quintile_lag[day]] - ticker_data[feat_quintile_lag[day + 1]]
            )
            ticker_data[feat_rsi_diff_abs[day]] = np.abs(ticker_data[feat_rsi_diff[day]])

        return ticker_data, feat_quintile_lag + feat_rsi_diff + feat_rsi_diff_abs
