import time
import json

from django.http import HttpResponse, HttpResponseRedirect
from django.views import View

from stockprediction.models import Stock
from stockprediction.tasks.collect_tweets_task import collect_tweets


class CollectTweetsEndpoint(View):
    """ View that acts as an endpoint to trigger Tweet collection """

    def get(self, request, stock=None):
        """ Handles GET requests and collects Stock Date

        If the argument `stock` isnt passed, it defaults to collect all
        Stocks data

        Parameters
        ----------
        request : Any
            The request object that view handles
        stock : str, optional
            The ticker of the stock to get the data for
        """

        if stock is None:
            stocks = Stock.objects.all()
            response_data = {}

            # Iterate over Stocks and collect Tweets
            for stock in stocks:
                collect_tweets(stock.ticker)

            response_data['response'] = 'Collected Tweets'
            return HttpResponse(json.dumps(response_data), status='200')

        else:
            stock = Stock.objects.get(ticker=stock)
            collect_tweets(stock.ticker)
            return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
