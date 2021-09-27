import datetime as dt
import os

from dateutil.relativedelta import relativedelta, FR
import numpy as np
import pandas as pd
import requests

from .common import download_data as download_data_generic
from .common import get_data as get_data_generic

FMP_API_KEY = os.environ.get('FMP_API_KEY')


def get_ticker_map():
    symbols = pd.read_json(f'''https://financialmodelingprep.com/api/v3/stock/list?apikey={FMP_API_KEY}''')
    stock_symbols = symbols[symbols['type'] == 'stock']
    stock_symbols['yahoo'] = stock_symbols['symbol']
    stock_symbols['bloomberg_ticker'] = np.where('.' not in stock_symbols['symbol'],
                                                 stock_symbols['symbol'] + ' US',
                                                 stock_symbols['symbol'])
    return stock_symbols[['bloomberg_ticker', 'yahoo']]


def download_ticker(ticker, start_epoch, end_epoch):
    """download data for a given ticker"""
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}'
    params = {
        'from': dt.datetime.fromtimestamp(start_epoch).strftime('%Y-%m-%d'),
        'to': dt.datetime.fromtimestamp(end_epoch).strftime('%Y-%m-%d'),
        'apikey': FMP_API_KEY
    }
    data = requests.get(
        url=url,
        params=params)
    data_json = data.json()
    df = pd.DataFrame(
        data_json['historical'],
        columns=['date', 'open', 'high', 'low', 'close', 'adjClose', 'volume'],
        dtype=np.float32,
    )
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['bloomberg_ticker'] = ticker
    df['provider'] = 'fmp'
    df['currency'] = 'USD'
    df.rename(columns={"adjClose": "adj_close"}, inplace=True)
    return ticker, df.drop_duplicates().dropna()


def download_data(db_dir, recreate=False, ticker_map=None):
    """download (missing) data for the tickers in the universe using the fmp API"""
    return download_data_generic(db_dir, download_ticker, recreate, ticker_map)


def get_data(
        db_dir,
        features_generators=None,
        last_friday=dt.datetime.today() - relativedelta(weekday=FR(-1)),
        target='target_20d',
        ticker_map=None
):
    """generate data set with the fmp API!"""
    return get_data_generic(
        db_dir,
        features_generators=features_generators,
        last_friday=last_friday,
        target=target,
        ticker_map=ticker_map)
