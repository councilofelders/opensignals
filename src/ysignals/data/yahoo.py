import shutil
import pandas as pd
import yfinance
import logging

from datetime import datetime
from dateutil.relativedelta import relativedelta, FR

logger = logging.getLogger(__name__)

AWS_BASE_URL='https://numerai-signals-public-data.s3-us-west-2.amazonaws.com'
SIGNALS_UNIVERSE=f'{AWS_BASE_URL}/latest_universe.csv'
SIGNALS_TICKER_MAP=f'{AWS_BASE_URL}/signals_ticker_map_w_bbg.csv'
SIGNALS_TARGETS=f'{AWS_BASE_URL}/signals_train_val_bbg.csv'

def get_tickers():
    ticker_universe = pd.read_csv(SIGNALS_UNIVERSE)
    ticker_map = pd.read_csv(SIGNALS_TICKER_MAP)
    ticker_map = ticker_map.dropna(subset=['yahoo'])
    ticker_map = ticker_map[ticker_map.bloomberg_ticker.isin(ticker_universe['bloomberg_ticker'])]
    logger.info(f'Number of eligible tickers: {ticker_map.shape[0]}')

    if ticker_map['yahoo'].duplicated().any():
        raise Exception(
            f'Found duplicated {ticker_map["yahoo"].duplicated().values().sum()}'
            ' yahoo tickers'
        )

    if ticker_map['bloomberg_ticker'].duplicated().any():
        raise Exception(
            f'Found duplicated {ticker_map["bloomberg_ticker"].duplicated().values().sum()}'
            ' bloomberg_ticker tickers'
        )

    return ticker_map


def get_ticker_data(db_dir):
    ticker_data = pd.DataFrame({
        'ticker' : pd.Series([], dtype='str'),
        'date' : pd.Series([], dtype='datetime64[ns]')
    })
    if len(list(db_dir.rglob('*.parquet'))) > 0:
        ticker_data = pd.read_parquet(db_dir)

    logger.info(f'Retrieving data for {ticker_data.ticker.unique().shape[0]} '
                 'tickers from the database')

    return ticker_data


def get_ticker_missing(
    ticker_data, ticker_map, last_friday = datetime.today() - relativedelta(weekday=FR(-1))
):
    tickers_available_data = ticker_data.groupby('ticker').agg({'date': [max, min]})
    tickers_available_data.columns = ['date_max', 'date_min']

    eligible_tickers_available_data = ticker_map.merge(
        tickers_available_data.reset_index(),
        left_on='bloomberg_ticker',
        right_on='ticker',
        how='left'
    )

    ticker_not_found = eligible_tickers_available_data.loc[
        eligible_tickers_available_data.ticker_y.isna(), ['bloomberg_ticker', 'yahoo']
    ]

    ticker_not_found['start'] = '2002-12-01'

    last_friday_20 = last_friday - relativedelta(days=20)
    tickers_outdated = eligible_tickers_available_data.loc[
        (
            (eligible_tickers_available_data.date_max < last_friday.strftime('%Y-%m-%d')) &
            (eligible_tickers_available_data.date_max > last_friday_20.strftime('%Y-%m-%d'))
        ),
        ['bloomberg_ticker', 'yahoo', 'date_max']
    ]

    tickers_outdated['start'] = (
        tickers_outdated['date_max'] + pd.DateOffset(1)
    ).dt.strftime('%Y-%m-%d')
    tickers_outdated.drop(columns=['date_max'], inplace=True)

    return pd.concat(
        [ticker_not_found, tickers_outdated]
    )


def get_data(
    db_dir,
    features_generators = [],
    last_friday = datetime.today() - relativedelta(weekday=FR(-1))
):
    ticker_data = get_ticker_data(db_dir)

    targets = pd.read_csv(SIGNALS_TARGETS)
    targets['date'] = pd.to_datetime(
        targets['friday_date'],
        format='%Y%m%d'
    )
    targets.rename(columns={'bloomberg_ticker': 'ticker'}, inplace=True)

    feature_names = []
    for features_generator in features_generators:
        ticker_data, feature_names_aux = features_generator.generate_features(ticker_data)
        feature_names.extend(feature_names_aux)

    # merge our feature data with Numerai targets
    ml_data = pd.merge(
        ticker_data, targets,
        on=['date', 'ticker']
    )

    # convert date to datetime and index on it
    ml_data = ml_data.set_index('date')

    # for training and testing we want clean, complete data only
    ml_data.dropna(inplace=True)
    # ensure we have only fridays
    ml_data = ml_data[ml_data.index.weekday == 4]
    # drop eras with under 50 observations per era
    ml_data = ml_data[ml_data.index.value_counts() > 50]

    # train test split
    train_data = ml_data[ml_data['data_type'] == 'train']
    test_data = ml_data[ml_data['data_type'] == 'validation']

    # generate live data
    date_string = last_friday.strftime('%Y-%m-%d')
    live_data = ticker_data[ticker_data.date == date_string].copy()

    # get data from the day before, for markets that were closed
    last_thursday = last_friday - relativedelta(days=1)
    thursday_date_string = last_thursday.strftime('%Y-%m-%d')
    thursday_data = ticker_data[ticker_data.date == thursday_date_string]

    # Only select tickers than aren't already present in live_data
    thursday_data = thursday_data[
        ~thursday_data.ticker.isin(live_data.ticker.values)
    ].copy()

    live_data = pd.concat([live_data, thursday_data])
    live_data = live_data.set_index('date')

    return train_data, test_data, live_data, feature_names


def download_data(db_dir, recreate = False):
    if recreate:
        logging.warn(f'Removing dataset {db_dir} to recreate it')
        shutil.rmtree(db_dir, ignore_errors=True)

    db_dir.mkdir(exist_ok=True)

    ticker_data = get_ticker_data(db_dir)
    ticker_map = get_tickers()
    ticker_missing = get_ticker_missing(ticker_data, ticker_map)

    n_ticker_missing = ticker_missing.shape[0]
    if n_ticker_missing <= 0:
        logger.info(f'Dataset up to date')
        return

    logger.info(f'Downloading missing data for {n_ticker_missing} tickers')

    ticker_missing_grouped = ticker_missing.groupby('start').apply(
        lambda x: ' '.join(x.yahoo.astype(str))
    )
    concat_dfs = []
    for start_date, tickers in ticker_missing_grouped.iteritems():
        temp_df = yfinance.download(tickers,
                                    start=start_date,
                                    threads=True)
        temp_df = temp_df.stack().reset_index().dropna()
        temp_df.columns = ['date', 'ticker', 'adj_close', 'close', 'hight', 'low', 'open', 'volume']
        temp_df['created_at'] = datetime.now()
        temp_df['volume'] = temp_df['volume'].astype('float64')

        # Yahoo Finance returning previous day in some situations (e.g. Friday in TelAviv markets)
        temp_df = temp_df[temp_df.date >= start_date]

        temp_df['ticker'] = temp_df['ticker'].map(
            dict(zip(ticker_map['yahoo'], ticker_map['bloomberg_ticker'])))

        concat_dfs.append(temp_df)

    df = pd.concat(concat_dfs)
    n_ticker_data = df.ticker.unique().shape[0]
    if n_ticker_data <= 0:
        logger.info(f'Dataset up to date')
        return

    logger.info(f'Storing data for {n_ticker_data} tickers')
    df.to_parquet(db_dir / f'{datetime.utcnow().timestamp()}.parquet', index=False)