from django.db import models
from .exchange import Exchange


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
