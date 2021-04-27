from django.views.generic import ListView

from stock_prediction.models import Stock


class StockList(ListView):
    model = Stock
    ordering = 'ticker'