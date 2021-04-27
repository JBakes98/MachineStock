import json

from django.http import HttpResponse
from django.views import View

from stock_prediction.models import Exchange, Stock


class AddStocksEndpoint(View):
    def get(self, request):
        if request.META.get('HTTP_X_APPENGINE_CRON'):  # Check request comes from GCP Cron
            exchange = [['NYSE', 'New York Stock Exchange'],
                        ['NASDAQ', 'Nasdaq']]

            stocks = [['AXP', 'American Express Co', 'NYSE'],
                      ['AMGN', 'Amgen Inc', 'NASDAQ'],
                      ['AAPL', 'Apple Inc', 'NASDAQ'],
                      ['BA', 'Boeing Co', 'NYSE'],
                      ['CAT', 'Caterpillar Inc', 'NYSE'],
                      ['CSCO', 'Cisco Systems Inc', 'NASDAQ'],
                      ['CVX', 'Chevron Corp', 'NYSE'],
                      ['GS', 'Goldman Sachs Group Inc', 'NYSE'],
                      ['HD', 'Home Depot Inc', 'NYSE'],
                      ['HON', 'Honeywell International Inc', 'NYSE'],
                      ['IBM', 'International Business Machines Corp', 'NYSE'],
                      ['INTC', 'Intel Corp', 'NASDAQ'],
                      ['JNJ', 'Johnson & Johnson', 'NYSE'],
                      ['KO', 'Coca-Cola Co', 'NYSE'],
                      ['JPM', 'JPMorgan Chase & Co', 'NYSE'],
                      ['MCD', "McDonald's Corp", 'NYSE'],
                      ['MMM', '3M Co', 'NYSE'],
                      ['MRK', 'Merck & Co Inc', 'NYSE'],
                      ['MSFT', 'Microsoft Corp', 'NASDAQ'],
                      ['NKE', 'Nike Inc', 'NYSE'],
                      ['PG', 'Procter & Gamble Co', 'NYSE'],
                      ['TRV', 'Travelers Companies Inc', 'NYSE'],
                      ['UNH', 'UnitedHealth Group Inc', 'NYSE'],
                      ['CRM', 'Salesforce.Com Inc', 'NYSE'],
                      ['VZ', 'Verizon Communications Inc', 'NYSE'],
                      ['V', 'Visa Inc', 'NYSE'],
                      ['WBA', 'Walgreens Boots Alliance Inc', 'NASDAQ'],
                      ['WMT', 'Walmart Inc', 'NYSE'],
                      ['DIS', 'Walt Disney Co', 'NYSE'],
                      ['DOW', 'Dow Inc', 'NYSE']]

            for i in exchange:
                Exchange.objects.update_or_create(symbol=i[0], name=i[1])

            for i in stocks:
                Stock.objects.update_or_create(ticker=i[0],
                                               name=i[1],
                                               exchange=Exchange.objects.get(symbol=i[2]))

        else:
            # Not a GCP Cron request so not authorised
            return HttpResponse(json.dumps({'response': 'Not authorised GCP service account'}), status="403")

        return HttpResponse(status='200')
