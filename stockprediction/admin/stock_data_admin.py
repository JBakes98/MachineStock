from django.contrib import admin
from stockprediction.models import StockData


class StockDataAdmin(admin.ModelAdmin):
    """ Class that defines the StockDate models admin properties """

    # Fields to display on the admin detail page
    fieldsets = (
        (None, {'fields': (
            'date',
            'stock',
            'open',
            'high',
            'low',
            'close',
            'adj_close',
            'volume',
            'change',
            'change_perc',
            'ma7',
            'ma21',
            'ema26',
            'ema12',
            'MACD',
            'sd20',
            'upper_band',
            'lower_band',
            'ema',
            'momentum',
            'log_momentum'
        )}),
    )

    # Fields to display on the admin list page
    list_display = ('date', 'stock', 'open', 'high', 'low', 'adj_close', 'change_perc', )
    list_filter = ('stock', 'date')  # Fields that can filter the list page items
    search_fields = ('stock', 'date')  # Fields that can be searched on the list page
    ordering = ('stock', 'date')  # Field to order the admin list by


admin.site.register(StockData, StockDataAdmin)  # Register the model and its admin class
