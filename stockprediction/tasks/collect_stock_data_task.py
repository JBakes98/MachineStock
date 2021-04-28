from django.conf import settings
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from stockprediction.models import Stock
from stockprediction.utils import dataframe_utils as df_utils
from stockprediction.utils import stock_utils


def collect_stock_data(ticker: str) -> pd.DataFrame:
    """ Method to collect Stock data from AlphaVantage

    Method collects the Stocks specified by the ticker data
    from the AlphaVantage API and formats the DataFrame and calculates
    the technical indicators for the adj close column.
    """

    # Initialise Alpha Vantage API from stored key
    ts = TimeSeries(key=settings.ALPHA_KEY, output_format='pandas')

    try:
        stock = Stock.objects.get(ticker=ticker)
    except Stock.DoesNotExist as e:
        raise e

    dataset, meta_data = ts.get_daily_adjusted(stock.ticker)
    dataset = df_utils.format_stock_dataset_for_db(dataset, stock.ticker)
    dataset = stock_utils.get_technical_indicators(dataset, 'adj close')

    return dataset
