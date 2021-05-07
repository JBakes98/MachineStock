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
    """ Model to represent a Stock """
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
        """ Get all of the Stocks related data and return it as a DataFrame"""
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
        """ Get all of the Stocks related data and return it as a DataFrame"""
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
        from stockprediction.utils import date_utils

        data = self.latest_data
        latest_weekday = date_utils.prev_weekday(datetime.date.today())
        return data.date.date() != latest_weekday

    def plot_technical_indicators(self, dataset: pd.DataFrame = None) -> Figure:
        from stockprediction.utils import chart_utils

        """ Create a Plotly Figure of the Stocks technical indicators

        This method creates a chart of the Stocks data on a Plotly figure
        in the output of a div for inclusion in templates.
        """

        dataset = self.get_data()
        plot = chart_utils.plot_tech_indicators(dataset, self.ticker)
        return plot

    def get_test_predictions(self):
        from stockprediction.machine_learning import StockMachineLearning
        from stockprediction.utils import dataframe_utils as df_utils

        dataset = self.get_ml_data()
        dataset = df_utils.format_stock_dataset_for_ml(dataset)
        ml = StockMachineLearning(dataset, self.ticker)
        return ml.plot_test_predictions()

    def get_future_predictions(self):
        from stockprediction.machine_learning import StockMachineLearning
        from stockprediction.utils import dataframe_utils as df_utils

        dataset = self.get_ml_data()
        dataset = df_utils.format_stock_dataset_for_ml(dataset)
        ml = StockMachineLearning(dataset, self.ticker)
        return ml.plot_future_predictions()

    def get_charts(self):
        from stockprediction.machine_learning import StockMachineLearning
        from stockprediction.utils import chart_utils
        from stockprediction.utils import dataframe_utils as df_utils

        dataset = self.get_data()
        ti_plot = chart_utils.plot_tech_indicators(dataset, self.ticker)

        ml_dataset = self.get_ml_data()
        ml_dataset = df_utils.format_stock_dataset_for_ml(dataset)
        ml = StockMachineLearning(ml_dataset, self.ticker)
        test_plot = ml.plot_test_predictions()
        fut_plot = ml.plot_future_predictions

        return ti_plot, test_plot, fut_plot
