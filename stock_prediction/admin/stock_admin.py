from django.contrib import admin
from stock_prediction.models import Stock


class StockAdmin(admin.ModelAdmin):
    fieldsets = (
        (None, {'fields': (
            'ticker',
            'name',
            'exchange',
        )}),
    )

    list_display = ('ticker', 'name', 'exchange', )
    list_filter = ('exchange', )
    search_fields = ('ticker', 'name')
    ordering = ('ticker', )


admin.site.register(Stock, StockAdmin)
