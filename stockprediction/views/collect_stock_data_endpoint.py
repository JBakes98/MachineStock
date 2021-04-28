import time

from django.http import HttpResponse
from django.views import View

from stockprediction.models import Stock
from stockprediction.tasks.collect_stock_data_task import collect_stock_data
from stockprediction.tasks.save_stock_data_task import save_stock_data


class CollectStockDataEndpoint(View):
    """ View that acts as an endpoint to trigger Stock data collection """
    def get(self, request):
            stocks = Stock.objects.all()
            i = 1

            try:
                for stock in stocks:
                    if i % 5 == 0:
                        time.sleep(60)

                    data = collect_stock_data(stock.ticker)
                    save_stock_data(data)

                    i += 1

                return HttpResponse(status='200')
            except ValueError:
                return HttpResponse(status='429')
