import numpy as np
from django.db import models
from .exchange import Exchange
from .stock_data import StockData


class Stock(models.Model):
    """
       A class used to represent a stockprediction

       ...

       Attributes
       ----------
       name : str
           the name of the stockprediction
       ticker : str
           the ticker of the stockprediction
       exchange : Exchange
           the exchange that the stockprediction is listed on

       Methods
       -------
       __str__(self)
           Prints the ticker attribute of the stockprediction when displayed

        get_data(self)
            Gets the data for the stockprediction as a pandas.Dataframe and cleans it
       """
    name = models.CharField(unique=True, max_length=255, blank=False, null=False)
    ticker = models.CharField(unique=True, max_length=4, blank=False, null=False)
    exchange = models.ForeignKey(Exchange, on_delete=models.CASCADE, blank=False, null=False)

    class Meta:
        verbose_name_plural = 'Stocks'

    def __str__(self):
        return self.ticker

    @property
    def latest_data(self):
        return self.stock_data.first()

    def get_data(self):
        return StockData.objects.filter(stock=self).order_by('-date')

    def plot_technical_indicators(self, dataset=None):
        # If dataset is not provided then collect dataset
        if dataset is None:
            try:
                dataset = self.get_data()
            except ValueError:
                return

        dataset.replace(0, np.nan, inplace=True)

        trace1 = {
            'name': self.ticker,
            'type': 'candlestick',
            'x': dataset['date'],
            'yaxis': 'y2',
            'low': dataset['low'],
            'high': dataset['high'],
            'open': dataset['open'],
            'close': dataset['close'],
        }
        trace2 = {
            "line": {"width": 1},
            "mode": "lines",
            "name": "Moving Average",
            "type": "scatter",
            "x": dataset['date'],
            "y": dataset['ma7'],
            "yaxis": "y2",
        }
        trace3 = {
            "name": "Volume",
            "type": "bar",
            "x": dataset['date'],
            "y": dataset['volume'],
            "yaxis": "y",

        }
        trace4 = {
            "line": {"width": 1},
            "name": "Bollinger Bands",
            "type": "scatter",
            "x": dataset['date'],
            "y": dataset['upper_band'],
            "yaxis": "y2",
            "marker": {"color": "#ccc"},
            "hoverinfo": "none",
            "legendgroup": "Bollinger Bands"
        }
        trace5 = {
            "line": {"width": 1},
            "type": "scatter",
            "x": dataset['date'],
            "y": dataset['lower_band'],
            "yaxis": "y2",
            "marker": {"color": "#ccc"},
            "hoverinfo": "none",
            "showlegend": False,
            "legendgroup": "Bollinger Bands"
        }
        data = ([trace1, trace2, trace3, trace4, trace5])

        plot_div = plot(Figure(data=data, layout=layout), output_type='div',)
        return plot_div
