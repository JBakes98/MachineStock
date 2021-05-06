import datetime

from django.views.generic import DetailView

from stockprediction.models import Stock
from stockprediction.utils import date_utils


class StockDetail(DetailView):
    """ View that displays details of a specific Stock """
    model = Stock
    queryset = Stock.objects.all()
    slug_field = 'ticker'
    slug_url_kwarg = 'ticker'

    def get_context_data(self, **kwargs):
        """ Method to get extra context

        Method collects extra context for the Stock such as
        charts or predictions.
        """

        context = super(StockDetail, self).get_context_data(**kwargs)
        latest_weekday = date_utils.prev_weekday(datetime.date.today())
        context['stock_data'] = self.object.latest_data
        # context['refresh'] = self.object.refresh
        context['ti_chart'] = self.object.plot_technical_indicators()

        return context
