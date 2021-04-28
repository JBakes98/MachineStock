import datetime

from django.views.generic import TemplateView

from stock_prediction.models import StockData
from stock_prediction.utils import date_utils


class IndexView(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)

        prev_weekday = date_utils.prev_weekday(datetime.date.today())
        context['data_date'] = StockData.objects.all().order_by('-date')[:1]
        context['latest_weekday'] = prev_weekday

        context['top_winners'] = StockData.objects.filter(
            date=prev_weekday
        ).order_by('-date', '-change_perc')[:5]

        context['top_losers'] = StockData.objects.filter(
            date=prev_weekday
        ).order_by('-date', 'change_perc')[:5]

        return context