import os

import datetime as dt
import time as _time
from typing import Tuple
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from opensignals.data.provider import Provider
from iexfinance import stocks
from iexfinance.utils.exceptions import IEXQueryError


class IEXCloud(Provider):
    """Implementation of a stock dividend APY that uses IEXCloud"""

    def __init__(self, iex_token: str = None, iex_api_version: str = 'iexcloud-sandbox', get_prices=True, get_dividends=True):
        # iexcloud package expects IEX_TOKEN and IEX_API_VERSION environment variables
        if "IEX_TOKEN" not in os.environ:
            if iex_token is None:
                raise AttributeError('Must provide iex_token or set IEX_TOKEN environment variable')
            os.environ["IEX_TOKEN"] = iex_token
        if "IEX_API_VERSION" not in os.environ:
            os.environ["IEX_API_VERSION"] = iex_api_version
        if not get_dividends and not get_prices:
            raise AttributeError("Must get dividends or prices")
        self.get_dividends = get_dividends
        self.get_prices = get_prices

    def download_ticker(self, ticker: str, start: dt.datetime, end: dt.datetime) -> Tuple[str, pd.DataFrame]:
        """download data for a given ticker"""

        # IEX cloud price data goes back 15 years max, so adjust start
        fifteen_years_ago = dt.datetime.now() - relativedelta(years=3)
        if start < fifteen_years_ago:
            start = fifteen_years_ago

        def empty_df() -> pd.DataFrame:
            return pd.DataFrame(columns=[
                "date", "bloomberg_ticker",
                "close", "frequency", "div_amount"])

        def empty_price_df() -> pd.DataFrame:
            return pd.DataFrame(columns=[
                "date", "bloomberg_ticker",
                "close"])

        def empty_dividend_df() -> pd.DataFrame:
            return pd.DataFrame(columns=[
                "date", "bloomberg_ticker",
                "frequency", "div_amount"])

        retries = 20
        tries = retries + 1
        backoff = 1
        while tries > 0:
            tries -= 1
            try:
                prices = empty_price_df()
                dividends = empty_dividend_df()
                if self.get_prices:
                    prices = stocks.get_historical_data(symbols=ticker, start=start, close_only=True)
                    prices = prices.resample('D').ffill()

                if self.get_dividends:
                    dividends = stocks.Stock(ticker).get_dividends(range='5y')
                    if dividends.empty and self.get_prices:
                        prices[['frequency', 'div_amount']] = np.nan
                        return ticker, prices
                    elif dividends.empty:
                        return ticker, empty_df()

                    dividends.index = pd.to_datetime(dividends.index)
                    dividends = dividends.resample('D').ffill()

                # resample both prices and dividends to daily time series
                # last known price/dividends is always assumed best truth, hence ffill
                joined = prices.join(dividends).ffill()

                df = pd.DataFrame({
                    "date": joined.index,
                    "bloomberg_ticker": ticker,
                    "close": joined['close'],
                    "frequency": joined['frequency'],
                    "div_amount": joined["amount"],
                    "provider": 'iexcloud'
                })

                return ticker, df.drop_duplicates().dropna()
            except IEXQueryError as iex:
                if iex.status == 404:
                    return ticker, empty_df()
                elif iex.status == 429:
                    _time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                else:
                    return ticker, empty_df()
            except Exception:
                return ticker, empty_df()

        return ticker, empty_df()
