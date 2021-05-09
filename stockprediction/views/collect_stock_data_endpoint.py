import time
import json

from django.http import HttpResponse, HttpResponseRedirect
from django.views import View

from stockprediction.models import Stock
from stockprediction.tasks.collect_stock_data_task import collect_stock_data
from stockprediction.tasks.save_stocks_task import save_stock_data


class CollectStockDataEndpoint(View):
    """ View that acts as an endpoint to trigger Stock data collection """

    def get(self, request, stock=None):
        if stock is None:
            stocks = Stock.objects.all()
            i = 1
            response_data = {}

            try:
                for stock in stocks:
                    if i % 5 == 0:
                        time.sleep(60)

                    data = collect_stock_data(stock.ticker)
                    save_stock_data(data)

                    i += 1

                response_data['response'] = 'Collected Stock data'

                return HttpResponse(json.dumps(response_data), status='200')
            except ValueError:
                response_data['response'] = 'Failed to collect data, try again later'
                return HttpResponse(status='429')
        else:
            stock = Stock.objects.get(ticker=stock)
            response_data = {}
            try:
                data = collect_stock_data(stock.ticker)
                save_stock_data(data)
                response_data['response'] = 'Collected Stock data'

                return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))

            except ValueError:
                response_data['response'] = 'Failed to collect data, try again later'
                return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
