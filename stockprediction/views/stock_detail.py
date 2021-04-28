from django.views.generic import DetailView

from stockprediction.models import Stock


class StockDetail(DetailView):
    model = Stock
    queryset = Stock.objects.all()
    slug_field = 'ticker'
    slug_url_kwarg = 'ticker'

    def get_context_data(self, **kwargs):
        context = super(StockDetail, self).get_context_data(**kwargs)
        context['ti_chart'] = self.object.plot_technical_indicators()

        return context
