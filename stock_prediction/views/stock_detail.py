from django.views.generic import DetailView

from stock_prediction.models import Stock


class StockDetail(DetailView):
    model = Stock
    queryset = Stock.objects.all()
    slug_field = 'ticker'
    slug_url_kwarg = 'ticker'
