from django.conf import settings
from alpha_vantage.timeseries import TimeSeries

from stock_prediction.models import Stock
from stock_prediction.utils import dataframe_utils as df_utils
from stock_prediction.utils import stock_utils


def collect_stock_data(ticker: str):
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
