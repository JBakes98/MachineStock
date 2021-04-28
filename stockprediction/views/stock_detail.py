from django.views.generic import DetailView

from stockprediction.models import Stock


class StockDetail(DetailView):
    model = Stock
    queryset = Stock.objects.all()
    slug_field = 'ticker'
    slug_url_kwarg = 'ticker'
