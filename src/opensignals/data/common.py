from concurrent import futures
from datetime import datetime, date, time
import time as _time
import logging
import shutil

from dateutil.relativedelta import relativedelta, FR
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

AWS_BASE_URL = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com'
SIGNALS_UNIVERSE = f'{AWS_BASE_URL}/latest_universe.csv'
SIGNALS_TICKER_MAP = f'{AWS_BASE_URL}/signals_ticker_map_w_bbg.csv'
SIGNALS_TARGETS = f'{AWS_BASE_URL}/signals_train_val_bbg.csv'


def empty_df():
    return pd.DataFrame(columns=[
        "date", "bloomberg_ticker",
        "open", "high", "low", "close",
        "adj_close", "volume", "currency", "provider"])


def get_tickers():
    ticker_map = pd.read_csv(SIGNALS_TICKER_MAP)
    ticker_map = ticker_map.dropna(subset=['yahoo'])
    logger.info(f'Number of eligible tickers: {ticker_map.shape[0]}')

    if ticker_map['yahoo'].duplicated().any():
        num = ticker_map["yahoo"].duplicated().values().sum()
        raise Exception(
            f'Found duplicated {num}'
            ' yahoo tickers'
        )

    if ticker_map['bloomberg_ticker'].duplicated().any():
        num = ticker_map["bloomberg_ticker"].duplicated().values().sum()
        raise Exception(
            f'Found duplicated {num}'
            ' bloomberg_ticker tickers'
        )

    return ticker_map


def get_ticker_data(db_dir):
    ticker_data = pd.DataFrame({
        'bloomberg_ticker': pd.Series([], dtype='str'),
        'date': pd.Series([], dtype='datetime64[ns]')
    })
    if len(list(db_dir.rglob('*.parquet'))) > 0:
        ticker_data = pd.read_parquet(db_dir)

    num = ticker_data.bloomberg_ticker.unique().shape[0]
    logger.info(f'Retrieving data for {num} tickers from the database')

    return ticker_data


def get_ticker_missing(
        ticker_data,
        ticker_map,
        last_friday=datetime.today() - relativedelta(weekday=FR(-1))):
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

    return pd.concat(
        [ticker_not_found, tickers_outdated]
    )


def get_data(
        db_dir,
        features_generators=None,
        last_friday=datetime.today() - relativedelta(weekday=FR(-1)),
        target='target_20d'):
    """generate data set"""

    if features_generators is None:
        features_generators = []

    ticker_data = get_ticker_data(db_dir)

    ticker_universe = pd.read_csv(SIGNALS_UNIVERSE)
    ticker_data = ticker_data[ticker_data.bloomberg_ticker.isin(
        ticker_universe['bloomberg_ticker'])]

    targets = pd.read_csv(SIGNALS_TARGETS)
    targets['date'] = pd.to_datetime(
        targets['friday_date'],
        format='%Y%m%d'
    )
    targets['target'] = targets[target]

    feature_names = []
    for features_generator in features_generators:
        ticker_data, feature_names_aux = features_generator.generate_features(ticker_data)
        feature_names.extend(feature_names_aux)

    # merge our feature data with Numerai targets
    ml_data = pd.merge(
        ticker_data, targets,
        on=['date', 'bloomberg_ticker'],
        how='left'
    )

    logger.info(f'Found {ml_data.target.isna().sum()}'
                'rows without target, filling with 0.5')
    ml_data['target'] = ml_data['target'].fillna(0.5)

    # convert date to datetime and index on it
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

    # generate live data
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

    return train_data, test_data, live_data, feature_names


def download_ticker_with_retry(ticker, start_epoch, end_epoch, download_ticker):
    """Download data with retry"""

    retries = 3
    tries = retries + 1
    backoff = 1

    while tries > 0:
        tries -= 1
        try:
            return download_ticker(ticker, start_epoch, end_epoch)
        except Exception:
            _time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    return ticker, empty_df()


def download_tickers(tickers, start, download_ticker):
    start_epoch = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
    end_epoch = int(datetime.combine(date.today(), time()).timestamp())

    pbar = tqdm(
        total=len(tickers),
        unit='tickers'
    )

    dfs = {}
    with futures.ThreadPoolExecutor() as executor:
        _futures = []
        for ticker in tickers:
            _futures.append(
                executor.submit(download_ticker_with_retry, ticker=ticker, start_epoch=start_epoch,
                                end_epoch=end_epoch, download_ticker=download_ticker)
            )

        for future in futures.as_completed(_futures):
            pbar.update(1)
            ticker, data = future.result()
            dfs[ticker] = data

    pbar.close()

    return pd.concat(dfs)


def download_data(db_dir, download_ticker, recreate=False):
    if recreate:
        logging.warning(f'Removing dataset {db_dir} to recreate it')
        shutil.rmtree(db_dir, ignore_errors=True)

    db_dir.mkdir(exist_ok=True)

    ticker_data = get_ticker_data(db_dir)
    ticker_map = get_tickers()
    ticker_missing = get_ticker_missing(ticker_data, ticker_map)

    n_ticker_missing = ticker_missing.shape[0]
    if n_ticker_missing <= 0:
        logger.info('Dataset up to date')
        return

    logger.info(f'Downloading missing data for {n_ticker_missing} tickers')

    ticker_missing_grouped = ticker_missing.groupby('start').apply(
        lambda x: ' '.join(x.yahoo.astype(str))
    )
    concat_dfs = []
    for start_date, tickers in ticker_missing_grouped.iteritems():
        temp_df = download_tickers(tickers.split(' '), start_date, download_ticker)

        # Yahoo Finance returning previous day in some situations
        # (e.g. Friday in TelAviv markets)
        temp_df = temp_df[temp_df.date >= start_date]
        if temp_df.empty:
            continue

        temp_df['created_at'] = datetime.now()
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
    df.to_parquet(db_dir / f'{datetime.utcnow().timestamp()}.parquet', index=False)
