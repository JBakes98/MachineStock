from django.views.generic import ListView

from stockprediction.models import Stock


class StockList(ListView):
    model = Stock
    ordering = 'ticker'