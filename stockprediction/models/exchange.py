from django.db import models


class Exchange(models.Model):
    """
    A class used to represent an Exchange

    ...

    Attributes
    ----------
    name : CharField()
        The name of the Exchange
    symbol : CharField()
        The symbol of the Exchange

    Methods
    -------
    __str__()
        Returns the symbol attribute of the class when called
    """

    name = models.CharField(unique=True, max_length=255, blank=False, null=False)
    symbol = models.CharField(unique=True, max_length=6, blank=False, null=False)

    class Meta:
        verbose_name_plural = 'Exchanges'

    def __str__(self):
        return self.symbol
