from django.views.generic import ListView

from stockprediction.models import Stock


class StockList(ListView):
    """ View that displays all Stocks ordered by ticker """
    model = Stock
    ordering = 'ticker'
