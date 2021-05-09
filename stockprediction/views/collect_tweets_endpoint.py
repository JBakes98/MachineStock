import time
import json

from django.http import HttpResponse, HttpResponseRedirect
from django.views import View

from stockprediction.models import Stock
from stockprediction.tasks import collect_tweets


class CollectTweetsEndpoint(View):
    """ View that acts as an endpoint to trigger Tweet collection """

    def get(self, request, stock=None):
        if stock is None:
            stocks = Stock.objects.all()
            response_data = {}

            for stock in stocks:
                collect_tweets(stock.ticker)
            response_data['response'] = 'Collected Tweets'
            return HttpResponse(json.dumps(response_data), status='200')

        else:
            stock = Stock.objects.get(ticker=stock)
            collect_tweets(stock.ticker)
            return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
