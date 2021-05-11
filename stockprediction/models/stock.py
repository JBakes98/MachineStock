# noinspection PyInterpreter
import datetime
from django.db import models
from django.db.models import F
import numpy as np
import pandas as pd
from plotly.offline import plot
from plotly.graph_objs import Figure
from .exchange import Exchange


class Stock(models.Model):
    """
    A class used to represent a Stock

    ...

    Attributes
    ----------
    name : CharField()
        The name of the Stock
    ticker : CharField()
        The ticker of the Stock
    exchange : ForeignKey()
        A Foreign Key relation to an Exchange

    Methods
    -------
    __str__()
        Returns the ticker attribute of the class when called
    latest_data()
        Returns the latest StockData object for the  Stock
    get_data()
        Gets the Stocks related StockData instances
    get_ml_data()
        Gets the machine learning dataset for Stocks
    refresh()
        Checks if the Stocks data is out of date and needs a refresh
    plot_technical_indicators()
        Plot the technical indicators of the Stock on a chart
    get_test_predictions()
        Get the machine learning test predictions chart for the Stock
    get_future_predictions()
        Get the machine learning future predictions chart for the Stock
    get_charts()
        Gets the test and future machine learning prediction charts
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
        """ Property that is the latest data for the Stock """

        return self.stock_data.first()

    def get_data(self) -> pd.DataFrame:
        """ Get all of the Stocks related data and return it as a DataFrame """

        from .stock_data import StockData

        dataset = pd.DataFrame.from_records(
            StockData.objects.filter(stock=self).select_related('stock').annotate(
                ticker=F('stock__ticker'), exchange=F('stock__exchange__symbol')
            ).values('date', 'ticker', 'exchange', 'high', 'low', 'open', 'close', 'adj_close', 'volume',
                     'dividend_amount',
                     'change', 'change_perc', 'ma7', 'ma21', 'ema26', 'ema12', 'MACD', 'sd20', 'upper_band',
                     'lower_band', 'ema', 'momentum', 'log_momentum')
        )

        # If there is not data found return a value error
        if dataset.empty:
            return ValueError
        return dataset

    def get_ml_data(self) -> pd.DataFrame:
        """ Get all of the StockData for machine learning use as a Dataframe """

        from .stock_data import StockData

        dataset = pd.DataFrame.from_records(
            StockData.objects.select_related('stock').annotate(
                ticker=F('stock__ticker'), exchange=F('stock__exchange__symbol')
            ).values('date', 'ticker', 'exchange', 'high', 'low', 'open', 'close', 'adj_close', 'volume',
                     'dividend_amount',
                     'change', 'change_perc', 'ma7', 'ma21', 'ema26', 'ema12', 'MACD', 'sd20', 'upper_band',
                     'lower_band', 'ema', 'momentum', 'log_momentum')
        )

        # If there is not data found return a value error
        if dataset.empty:
            return ValueError
        return dataset

    def refresh(self):
        """ Checks if the Stocks data is out of date and needs a refresh """

        from stockprediction.utils import date_utils

        data = self.latest_data
        latest_weekday = date_utils.prev_weekday(datetime.date.today())
        return (data.date.date() != latest_weekday) and (data.date.date() != datetime.date.today())

    def plot_technical_indicators(self) -> Figure:
        """ Create a Plotly Figure of the Stocks technical indicators

        This method creates a chart of the Stocks data on a Plotly figure
        in the output of a div for inclusion in templates.
        """

        dataset = self.get_data()

        # Replace 0 with Nan so indicators such as ma that don't have a value
        # until 7 days of data don't display inaccurate data
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

        layout = {
            "xaxis": {
                "title": "Adjusted Close",
                "rangeselector": {
                    "x": 0,
                    "y": 0.9,
                    "font": {"size": 13},

                    "buttons": [
                        {
                            "step": "all",
                            "count": 1,
                            "label": "reset"
                        },
                        {
                            "step": "month",
                            "count": 3,
                            "label": "3 mo",
                            "stepmode": "backward"
                        },
                        {
                            "step": "month",
                            "count": 1,
                            "label": "1 mo",
                            "stepmode": "backward"
                        },
                        {"step": "all"}
                    ]
                }
            },
            "yaxis": {
                "domain": [0, 0.2],
                "showticklabels": False,
            },
            "legend": {
                "x": 0.3,
                "y": 0.9,
                "yanchor": "bottom",
                "orientation": "h"
            },
            "margin": {
                "b": 30,
                "l": 30,
                "r": 30,
                "t": 30,
            },
            "yaxis2": {"domain": [0.2, 0.8]},
            "plot_bgcolor": "rgb(250, 250, 250)"
        }

        plot_div = plot(Figure(data=data, layout=layout), output_type='div')

        return plot_div

    def get_test_predictions(self):
        """ Create a Plotly Figure of the Stocks machine learning test predictions

        This method creates a chart of the machine learning test predictions
        on a Plotly figure in the output of a div for inclusion in templates.
        """

        from stockprediction.machine_learning import StockMachineLearning
        from stockprediction.utils import dataframe_utils as df_utils

        dataset = self.get_ml_data()
        dataset = df_utils.format_stock_dataset_for_ml(dataset)
        ml = StockMachineLearning(dataset, self.ticker)
        return ml.plot_test_predictions()

    def get_future_predictions(self):
        """ Create a Plotly Figure of the Stocks machine learning future predictions

        This method creates a chart of the machine learning future predictions
        on a Plotly figure in the output of a div for inclusion in templates.
        """

        from stockprediction.machine_learning import StockMachineLearning
        from stockprediction.utils import dataframe_utils as df_utils

        dataset = self.get_ml_data()
        dataset = df_utils.format_stock_dataset_for_ml(dataset)
        ml = StockMachineLearning(dataset, self.ticker)
        return ml.plot_future_predictions()

    def get_charts(self):
        """ Gets the Stocks three charts

        This method gets the three stocks charts and returns all three.
        """

        from stockprediction.machine_learning import StockMachineLearning
        from stockprediction.utils import dataframe_utils as df_utils

        dataset = self.get_data()
        ti_plot = self.plot_technical_indicators()

        ml_dataset = self.get_ml_data()
        ml_dataset = df_utils.format_stock_dataset_for_ml(dataset)
        ml = StockMachineLearning(ml_dataset, self.ticker)
        test_plot = ml.plot_test_predictions()
        fut_plot = ml.plot_future_predictions

        return ti_plot, test_plot, fut_plot
