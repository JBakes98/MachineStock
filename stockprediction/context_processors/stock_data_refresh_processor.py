import datetime

from stockprediction.models import StockData
from stockprediction.utils import date_utils


def stock_data_refresh_processor(request):
    """ Context processor that checks if StockData needs a refresh """

    data_date = StockData.objects.all().order_by('-date').first()  # Select first latest data
    data_date = data_date.date  # Assign to the date attribute of the model
    latest_weekday = date_utils.prev_weekday(datetime.date.today())  # Get the previous market day

    return {'data_refresh': data_date.date() != latest_weekday} # Return context bool if data is out of date
