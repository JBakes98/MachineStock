from django.contrib import admin
from stockprediction.models import StockData


class StockDataAdmin(admin.ModelAdmin):
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

    list_display = ('date', 'stock', 'open', 'high', 'low', 'adj_close', 'change_perc', )
    list_filter = ('stock', 'date')
    search_fields = ('stock', 'date')
    ordering = ('stock', 'date')


admin.site.register(StockData, StockDataAdmin)
