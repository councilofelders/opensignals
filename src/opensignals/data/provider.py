from abc import ABC, abstractmethod
from concurrent import futures
import datetime as dt
import logging
import pathlib
import shutil
from typing import Optional, List, Tuple

from dateutil.relativedelta import relativedelta, FR
import pandas as pd
from tqdm import tqdm

from opensignals import features


logger = logging.getLogger(__name__)

AWS_BASE_URL = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com'
SIGNALS_UNIVERSE = f'{AWS_BASE_URL}/latest_universe.csv'
SIGNALS_TICKER_MAP = f'{AWS_BASE_URL}/signals_ticker_map_w_bbg.csv'
SIGNALS_TARGETS = f'{AWS_BASE_URL}/signals_train_val_bbg.csv'


class Provider(ABC):
    """Common base class for (daily) stock price data"""

    @staticmethod
    def get_tickers() -> pd.DataFrame:
        ticker_map = pd.read_csv(SIGNALS_TICKER_MAP)
        ticker_map = ticker_map.dropna(subset=['yahoo'])
        logger.info(f'Number of eligible tickers: {ticker_map.shape[0]}')

        if ticker_map['yahoo'].duplicated().any():
            num = ticker_map["yahoo"].duplicated().values.sum()
            raise Exception(f'Found duplicated {num} yahoo tickers')

        if ticker_map['bloomberg_ticker'].duplicated().any():
            num = ticker_map["bloomberg_ticker"].duplicated().values.sum()
            raise Exception(f'Found duplicated {num} bloomberg_ticker tickers')

        return ticker_map

    @staticmethod
    def get_ticker_data(db_dir: pathlib.Path) -> pd.DataFrame:
        ticker_data = pd.DataFrame({
            'bloomberg_ticker': pd.Series([], dtype='str'),
            'date': pd.Series([], dtype='datetime64[ns]')
        })
        if len(list(db_dir.rglob('*.parquet'))) > 0:
            ticker_data = pd.read_parquet(db_dir)

        num = ticker_data.bloomberg_ticker.unique().shape[0]
        logger.info(f'Retrieving data for {num} tickers from the database')

        return ticker_data

    @staticmethod
    def get_ticker_missing(ticker_data: pd.DataFrame,
                           ticker_map: pd.DataFrame,
                           last_friday: Optional[dt.datetime] = None) -> pd.DataFrame:
        if last_friday is None:
            last_friday = dt.datetime.today() - relativedelta(weekday=FR(-1))
        tickers_available_data = ticker_data.groupby('bloomberg_ticker').agg({'date': [max, min]})
        tickers_available_data.columns = ['date_max', 'date_min']

        eligible_tickers_available_data = ticker_map.merge(
            tickers_available_data.reset_index(),
            on='bloomberg_ticker',
            how='left'
        )

        ticker_not_found = eligible_tickers_available_data.loc[
            eligible_tickers_available_data.date_max.isna(), ['bloomberg_ticker', 'yahoo']
        ]

        ticker_not_found['start'] = '2002-12-01'

        last_friday_52 = last_friday - relativedelta(weeks=52)
        tickers_outdated = eligible_tickers_available_data.loc[
            (
                    (eligible_tickers_available_data.date_max < last_friday.strftime('%Y-%m-%d')) &
                    (eligible_tickers_available_data.date_max > last_friday_52.strftime('%Y-%m-%d'))
            ),
            ['bloomberg_ticker', 'yahoo', 'date_max']
        ]

        tickers_outdated['start'] = (
                tickers_outdated['date_max'] + pd.DateOffset(1)
        ).dt.strftime('%Y-%m-%d')
        tickers_outdated.drop(columns=['date_max'], inplace=True)

        return pd.concat([ticker_not_found, tickers_outdated])  # type: ignore

    @staticmethod
    def get_live_data(ticker_data: pd.DataFrame, last_friday: dt.date) -> pd.DataFrame:
        date_string = last_friday.strftime('%Y-%m-%d')
        live_data = ticker_data[ticker_data.date == date_string].copy()

        # get data from the day before, for markets that were closed
        last_thursday = last_friday - relativedelta(days=1)
        thursday_date_string = last_thursday.strftime('%Y-%m-%d')
        thursday_data = ticker_data[ticker_data.date == thursday_date_string]

        # Only select tickers than aren't already present in live_data
        thursday_data = thursday_data[
            ~thursday_data.bloomberg_ticker.isin(live_data.bloomberg_ticker.values)
        ].copy()

        live_data = pd.concat([live_data, thursday_data])
        live_data = live_data.set_index('date')
        return live_data  # type: ignore

    @staticmethod
    def get_train_test_data(ticker_data: pd.DataFrame,
                            feature_names: List[str],
                            targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """merge our feature data with Numerai targets"""
        ml_data = pd.merge(
            ticker_data, targets,
            on=['date', 'bloomberg_ticker'],
            how='left'
        )

        logger.info(f'Found {ml_data.target.isna().sum()}'
                    'rows without target, filling with 0.5')
        ml_data['target'] = ml_data['target'].fillna(0.5)

        ml_data = ml_data.set_index('date')

        # for training and testing we want clean, complete data only
        ml_data = ml_data.dropna(subset=feature_names)
        # ensure we have only fridays
        ml_data = ml_data[ml_data.index.weekday == 4]
        # drop eras with under 50 observations per era
        ml_data = ml_data[ml_data.index.value_counts() > 50]

        # train test split
        train_data = ml_data[ml_data['data_type'] == 'train']
        test_data = ml_data[ml_data['data_type'] == 'validation']
        return train_data, test_data

    def get_data(self,
                 db_dir: pathlib.Path,
                 features_generators: Optional[List[features.FeatureGenerator]] = None,
                 last_friday: Optional[dt.datetime] = None,
                 target: str = 'target_20d',
                 feature_prefix: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """generate data set"""
        if last_friday is None:
            last_friday = dt.datetime.today() - relativedelta(weekday=FR(-1))
        if features_generators is None:
            features_generators = []

        ticker_data = self.get_ticker_data(db_dir)

        targets = pd.read_csv(SIGNALS_TARGETS)
        targets['date'] = pd.to_datetime(
            targets['friday_date'],
            format='%Y%m%d'
        )
        targets['target'] = targets[target]

        feature_names = []
        for features_generator in features_generators:
            ticker_data, feature_names_aux = features_generator.generate_features(ticker_data, feature_prefix)
            feature_names.extend(feature_names_aux)

        train_data, test_data = Provider.get_train_test_data(ticker_data, feature_names, targets)

        # generate live data
        live_data = Provider.get_live_data(ticker_data, last_friday)

        return train_data, test_data, live_data, feature_names

    def download_tickers(self, tickers: pd.DataFrame, start: str) -> pd.DataFrame:
        start_date = dt.datetime.strptime(start, '%Y-%m-%d')
        end_date = dt.datetime.combine(dt.date.today(), dt.time())

        pbar = tqdm(total=len(tickers), unit='tickers')

        dfs = {}
        with futures.ThreadPoolExecutor() as executor:
            _futures = []
            for ticker in tickers:
                _futures.append(
                    executor.submit(self.download_ticker, ticker=ticker, start=start_date, end=end_date)
                )

            for future in futures.as_completed(_futures):
                pbar.update(1)
                ticker, data = future.result()
                dfs[ticker] = data

        pbar.close()

        return pd.concat(dfs)

    def download_data(self, db_dir: pathlib.Path, recreate: bool = False) -> None:
        if recreate:
            logging.warning(f'Removing dataset {db_dir} to recreate it')
            shutil.rmtree(db_dir, ignore_errors=True)

        db_dir.mkdir(exist_ok=True)

        ticker_data = self.get_ticker_data(db_dir)
        ticker_map = self.get_tickers()
        ticker_missing = self.get_ticker_missing(ticker_data, ticker_map)

        n_ticker_missing = ticker_missing.shape[0]
        if n_ticker_missing <= 0:
            logger.info('Dataset up to date')
            return

        logger.info(f'Downloading missing data for {n_ticker_missing} tickers')

        ticker_missing_grouped = ticker_missing.groupby('start').apply(
            lambda x: ' '.join(x.yahoo.astype(str))
        )
        concat_dfs = []
        for start, tickers in ticker_missing_grouped.iteritems():
            temp_df = self.download_tickers(tickers.split(' '), start=start)
            start_date = dt.datetime.strptime(start, '%Y-%m-%d')

            # Yahoo Finance returning previous day in some situations
            # (e.g. Friday in TelAviv markets)
            temp_df = temp_df[temp_df.date >= start_date]
            if temp_df.empty:
                continue

            temp_df['created_at'] = dt.datetime.now()
            temp_df['volume'] = temp_df['volume'].astype('float64')
            temp_df['bloomberg_ticker'] = temp_df['bloomberg_ticker'].map(
                dict(zip(ticker_map['yahoo'], ticker_map['bloomberg_ticker'])))

            concat_dfs.append(temp_df)

        if len(concat_dfs) == 0:
            logger.info('Dataset up to date')
            return

        df = pd.concat(concat_dfs)
        n_ticker_data = df.bloomberg_ticker.unique().shape[0]
        if n_ticker_data <= 0:
            logger.info('Dataset up to date')
            return

        logger.info(f'Storing data for {n_ticker_data} tickers')
        df.to_parquet(db_dir / f'{dt.datetime.utcnow().timestamp()}.parquet', index=False)
        return

    @abstractmethod
    def download_ticker(self, ticker: str, start: dt.datetime, end: dt.datetime) -> Tuple[str, pd.DataFrame]:
        pass
