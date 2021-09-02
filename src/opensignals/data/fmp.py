import datetime as dt
import numpy as np
import pandas as pd
import time as _time
import requests
import os

from .common import download_data as download_data_generic
from .common import get_data as get_data_generic
from dateutil.relativedelta import relativedelta, FR


FMP_API_KEY = os.environ['FMP_API_KEY']


def download_ticker(ticker, start_epoch, end_epoch):
    def empty_df():
        return pd.DataFrame(columns=[
            "date", "bloomberg_ticker",
            "open", "high", "low", "close",
            "adj_close", "volume", "currency", "provider"])

    retries = 3
    tries = retries + 1
    backoff = 1
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}'
    params = {
        'from': dt.datetime.fromtimestamp(start_epoch).strftime('%Y-%m-%d'),
        'to': dt.datetime.fromtimestamp(end_epoch).strftime('%Y-%m-%d'),
        'apikey': FMP_API_KEY
    }

    while tries > 0:
        tries -= 1
        try:
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

        except Exception as e:
            _time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    return ticker, empty_df()


def download_data(db_dir, recreate=False):
    return download_data_generic(db_dir, download_ticker, recreate)

def get_data(
        db_dir,
        features_generators = [],
        last_friday=dt.datetime.today() - relativedelta(weekday=FR(-1)),
        target='target'
): return get_data_generic(
    db_dir,
    features_generators=features_generators,
    last_friday=last_friday,
    target=target)
