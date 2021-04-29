import datetime

from stockprediction.models import StockData
from stockprediction.utils import date_utils


def stock_data_refresh_processor(request):
    data_date = StockData.objects.all().order_by('-date')[:1]
    latest_weekday = date_utils.prev_weekday(datetime.date.today())
    return {'data_refresh': data_date.date == latest_weekday}
