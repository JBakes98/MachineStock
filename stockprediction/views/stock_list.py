from django.views.generic import ListView

from stockprediction.models import Stock,Tweet


class StockList(ListView):
    """ View that displays all Stocks ordered by ticker """
    model = Stock
    ordering = 'ticker'

    def get_context_data(self, **kwargs):
        context = super(StockList, self).get_context_data(**kwargs)
        context['tweets'] = Tweet.objects.all()[:50]

        return context

