from decimal import Decimal

from django.db import models
from .stock import Stock


class StockData(models.Model):
    """
       A class used to represent a stocks past data

       ...

       Attributes
       ----------
       id : BigAutoField
           the id for the specific data, a big int due to the number of
           stock periods that may be stored from over multiple years
       date : DateTimeField
           the date and time this  data is for
       stock : Stock
           the Stock object this data is for

       Methods
       -------
       __str__(self)
           Prints the ticker attribute of the stock when displayed

        get_data(self)
            Gets the data for the stock as a pandas.Dataframe and cleans it
       """
    id = models.BigAutoField(primary_key=True)
    date = models.DateTimeField()
    stock = models.ForeignKey(Stock, related_name='stock_data', on_delete=models.CASCADE)

    # Data retrieved from Alpha Vantage API
    high = models.DecimalField(max_digits=15, decimal_places=4, default=Decimal(0.00))
    low = models.DecimalField(max_digits=15, decimal_places=4, default=Decimal(0.00))
    open = models.DecimalField(max_digits=15, decimal_places=4, default=Decimal(0.00))
    close = models.DecimalField(max_digits=15, decimal_places=4, default=Decimal(0.00))
    adj_close = models.DecimalField(max_digits=15, decimal_places=4, default=Decimal(0.00))
    volume = models.BigIntegerField(blank=True, null=True)

    # These values are calculated for the
    change = models.DecimalField(max_digits=15, decimal_places=4, default=Decimal(0.00))
    change_perc = models.DecimalField(max_digits=15, decimal_places=4, default=Decimal(0.00))

    # Technical indicators for the stock data
    ma7 = models.DecimalField(max_digits=15, decimal_places=4)
    ma21 = models.DecimalField(max_digits=15, decimal_places=4)
    ema26 = models.DecimalField(max_digits=15, decimal_places=4)
    ema12 = models.DecimalField(max_digits=15, decimal_places=4)
    MACD = models.DecimalField(max_digits=15, decimal_places=4)
    sd20 = models.DecimalField(max_digits=15, decimal_places=4)
    upper_band = models.DecimalField(max_digits=15, decimal_places=4)
    lower_band = models.DecimalField(max_digits=15, decimal_places=4)
    ema = models.DecimalField(max_digits=15, decimal_places=4)
    momentum = models.DecimalField(max_digits=15, decimal_places=4)
    log_momentum = models.DecimalField(max_digits=15, decimal_places=4)

    class Meta:
        verbose_name_plural = 'Stock Price Data'
        ordering = ['stock', '-date']

        indexes = [
            models.Index(fields=['date', 'stock'])
        ]

        constraints = [
            models.UniqueConstraint(fields=['stock', 'date'], name='stocks_day_data')
        ]

    def __str__(self):
        return '{}: {}'.format(self.stock, self.date)
