import time

from django.http import HttpResponse
from django.views import View

from stock_prediction.models import Stock
from stock_prediction.tasks.collect_stock_data_task import collect_stock_data
from stock_prediction.tasks.save_stock_data_task import save_stock_data


class CollectStockDataEndpoint(View):
    def get(self, request):
        if request.META.get('HTTP_X_APPENGINE_CRON'):  # Check request comes from GCP Cron
            stocks = Stock.objects.all()
            i = 1
            for stock in stocks:
                if i % 5 == 0:
                    time.sleep(60)

                data = collect_stock_data(stock.ticker)
                save_stock_data(data)

                i += 1
        else:
            # Not a GCP Cron request so not authorised
            return HttpResponse({'response': 'Not authorised GCP service account'}, status="403")

        return HttpResponse(status='200')
