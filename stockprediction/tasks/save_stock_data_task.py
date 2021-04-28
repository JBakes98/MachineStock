from django.db import IntegrityError

import pandas as pd

from stockprediction.models import Stock, StockData


def save_stock_data(dataset: pd.DataFrame):
    stock = Stock.objects.get(ticker=dataset['stock'].iloc[0])
    for row in dataset.itertuples():
        try:
            StockData.objects.create(
                stock=stock,
                date=getattr(row, 'date'),
                open=getattr(row, 'open'),
                high=getattr(row, 'high'),
                low=getattr(row, 'low'),
                close=getattr(row, 'close'),
                adj_close=getattr(row, '_6'),
                volume=getattr(row, 'volume'),
                change=getattr(row, 'change'),
                change_perc=getattr(row, 'change_perc'),
                ma7=getattr(row, 'ma7'),
                ma21=getattr(row, 'ma21'),
                ema26=getattr(row, 'ema26'),
                ema12=getattr(row, 'ema12'),
                MACD=getattr(row, 'MACD'),
                sd20=getattr(row, 'sd20'),
                upper_band=getattr(row, 'upper_band'),
                lower_band=getattr(row, 'lower_band'),
                ema=getattr(row, 'ema'),
                momentum=getattr(row, 'momentum'),
                log_momentum=getattr(row, 'log_momentum'))
        except IntegrityError:
            pass
