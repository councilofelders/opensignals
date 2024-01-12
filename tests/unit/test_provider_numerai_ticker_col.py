import unittest
from unittest.mock import patch
import datetime as dt
from opensignals.data.yahoo import Yahoo


class TestProviderNumeraiTickerCol(unittest.TestCase):

    @patch('opensignals.data.provider.dt.datetime', wraps=dt.datetime)
    def test_numerai_ticker_col_before_datetime(self, mock_datetime):
        mock_datetime.now.return_value = dt.datetime(2024, 1, 23, 12, 0, 0, tzinfo=dt.timezone.utc)

        provider = Yahoo()
        self.assertEqual(provider.numerai_ticker_col, 'bloomberg_ticker')

    @patch('opensignals.data.provider.dt.datetime', wraps=dt.datetime)
    def test_numerai_ticker_col_on_datetime(self, mock_datetime):
        # Set the current time to exactly 2024-01-23 13:00:00 UTC
        mock_datetime.now.return_value = dt.datetime(2024, 1, 23, 13, 0, 0, tzinfo=dt.timezone.utc)

        provider = Yahoo()
        self.assertEqual(provider.numerai_ticker_col, 'numerai_ticker')

    @patch('opensignals.data.provider.dt.datetime', wraps=dt.datetime)
    def test_numerai_ticker_col_after_datetime(self, mock_datetime):
        # Set the current time to after 2024-01-23 13:00:00 UTC
        mock_datetime.now.return_value = dt.datetime(2024, 1, 23, 14, 0, 0, tzinfo=dt.timezone.utc)

        provider = Yahoo()
        self.assertEqual(provider.numerai_ticker_col, 'numerai_ticker')


if __name__ == '__main__':
    unittest.main()
