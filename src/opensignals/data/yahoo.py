import datetime as dt
import random

from dateutil.relativedelta import relativedelta, FR
import numpy as np
import pandas as pd
import requests

from opensignals import utils
from .common import download_data as download_data_generic
from .common import empty_df
from .common import get_data as get_data_generic


def download_ticker(ticker, start_epoch, end_epoch):
    """download data for a given ticker"""
    url = f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker}'
    user_agent = random.choice(utils.USER_AGENTS)
    params = dict(
        period1=start_epoch,
        period2=end_epoch,
        interval='1d',
        events='div,splits',
    )
    data = requests.get(
        url=url,
        params=params,
        headers={'User-Agent': user_agent}
    )
    data_json = data.json()
    quotes = data_json["chart"]["result"][0]
    if "timestamp" not in quotes:
        return ticker, empty_df()

    timestamps = quotes["timestamp"]
    ohlc = quotes["indicators"]["quote"][0]
    volumes = ohlc["volume"]
    opens = ohlc["open"]
    closes = ohlc["close"]
    lows = ohlc["low"]
    highs = ohlc["high"]

    adjclose = closes
    if "adjclose" in quotes["indicators"]:
        adjclose = quotes["indicators"]["adjclose"][0]["adjclose"]

    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s").normalize(),
        "bloomberg_ticker": ticker,
        "open": np.array(opens, dtype='float32'),
        "high": np.array(highs, dtype='float32'),
        "low": np.array(lows, dtype='float32'),
        "close": np.array(closes, dtype='float32'),
        "adj_close": np.array(adjclose, dtype='float32'),
        "volume": np.array(volumes, dtype='float32'),
        "currency": quotes['meta']['currency'],
        "provider": 'yahoo'
    })

    return ticker, df.drop_duplicates().dropna()


def download_data(db_dir, recreate=False, ticker_map=None):
    """download (missing) data for the tickers in the universe using the Yahoo! Finance API"""
    return download_data_generic(db_dir, download_ticker, recreate, ticker_map=ticker_map)


def get_data(
        db_dir,
        features_generators=None,
        last_friday=dt.datetime.today() - relativedelta(weekday=FR(-1)),
        target='target_20d',
        ticker_map=None
):
    """generate data set with the Yahoo! Finance API!"""
    return get_data_generic(
        db_dir,
        features_generators=features_generators,
        last_friday=last_friday,
        target=target,
        ticker_map=ticker_map)
