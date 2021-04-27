from django.db import models
from .exchange import Exchange


class Stock(models.Model):
    """
       A class used to represent a stock_prediction

       ...

       Attributes
       ----------
       name : str
           the name of the stock_prediction
       ticker : str
           the ticker of the stock_prediction
       exchange : Exchange
           the exchange that the stock_prediction is listed on

       Methods
       -------
       __str__(self)
           Prints the ticker attribute of the stock_prediction when displayed

        get_data(self)
            Gets the data for the stock_prediction as a pandas.Dataframe and cleans it
       """
    name = models.CharField(unique=True, max_length=255, blank=False, null=False)
    ticker = models.CharField(unique=True, max_length=4, blank=False, null=False)
    exchange = models.ForeignKey(Exchange, on_delete=models.CASCADE, blank=False, null=False)

    class Meta:
        verbose_name_plural = 'Stocks'

    def __str__(self):
        return self.ticker
