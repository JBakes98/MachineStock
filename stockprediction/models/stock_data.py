from decimal import Decimal

from django.db import models
from .stock import Stock


class StockData(models.Model):
    """
    A class used to represent Stock Data

    ...

    Attributes
    ----------
    id : BigAutoField()
        The id of the stock,
    date : DateTimefield()
        The date the data is for
    stock : ForeignKey()
        A Foreign Key relation to a Stock the data is for
    high : DecimalField()
        The highest price of the Stock on a date
    low : DecimalField()
        The lowest price of a Stock on a date
    open : DecimalField()
        The open price of a Stock on a date
    close : DecimalField()
        The close price of a Stock on a date
    adj_close : DecimalField()
        The adjusted close price of a Stock on a date
    volume : BigIntergerField()
        The volume of shares of a Stock traded on a date
    dividend_amount : DecimalField()
        The dividend amount for a Stock on a date
    change : DecimalField()
        The change between the adj_close to the date before this date
    change_perc : DecimalField()
        The change between the adj_close to the date before this date as a percent
    ma7 : DecimalField()
        The 7 day moving average value on a date
    ma21 : DecimalField()
        The 21 day moving average value on a date
    ema26 : DecimalField()
        The 26 day exponential moving average value on a date
    ema12 : DecimalField()
        The 12 day exponential moving average value on a date
    MACD : DecimalField()
        The Moving Average Convergence Divergence value on a date
    sd20 : DecimalField()
        The 20 day moving average used for bollinger bands
    upper_band : DecimalField()
        The value of the upper band of bollinger bands on a date
    lower_band  : DecimalField()
        The value of the lower band of bollinger bands on a date

    Methods
    -------
    __str__()
        Returns a string of the related Stock the data is for and the date of the data
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
    dividend_amount = models.DecimalField(max_digits=15, decimal_places=4, default=Decimal(0.00))

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
