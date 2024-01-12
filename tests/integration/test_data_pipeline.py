import unittest
from pathlib import Path

from opensignals.data.yahoo import Yahoo
from opensignals.features import RSI, SMA


class TestDataPipeline(unittest.TestCase):

    def test_yahoo_rsi_data(self):
        db_dir = Path('db')

        yahoo = Yahoo()
        yahoo.download_data(db_dir)

        features_generators = [
            RSI(num_days=5, interval=14, variable='adj_close'),
            RSI(num_days=5, interval=21, variable='adj_close'),
            SMA(num_days=5, interval=14, variable='adj_close'),
            SMA(num_days=5, interval=21, variable='adj_close'),
        ]

        train, test, live, feature_names = yahoo.get_data(db_dir,
                                                          features_generators=features_generators,
                                                          feature_prefix='feature')

        # check that all the features are in each dataset
        for df in [train, test, live]:
            for feature_name in feature_names:
                self.assertTrue(feature_name in df.columns)

        # live dataset should be at least 3000 tickers
        self.assertTrue(live.shape[0] > 3000)

        # training dataset should be 859750 rows
        self.assertTrue(train.shape[0] >= 859750)

        # test dataset should be at least 1830145 rows
        self.assertTrue(test.shape[0] >= 1830145)

        # TODO: what assertions for data?
        #       feature_names are expected...
        #       live df > 3000 rows?
        #       training df should always be same length (859750)
        #       test df should be at least (1830145)
        print(train)