from django.db import models


class Exchange(models.Model):
    """ Model to represent a Stock Exchange"""
    name = models.CharField(unique=True, max_length=255, blank=False, null=False)
    symbol = models.CharField(unique=True, max_length=6, blank=False, null=False)

    class Meta:
        verbose_name_plural = 'Exchanges'

    def __str__(self):
        return self.symbol
