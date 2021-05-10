from django.contrib import admin
from stockprediction.models import Stock


class StockAdmin(admin.ModelAdmin):
    """ Class that defines the Stock models admin properties """

    # Fields to display on the admin detail page
    fieldsets = (
        (None, {'fields': (
            'ticker',
            'name',
            'exchange',
        )}),
    )

    list_display = ('ticker', 'name', 'exchange', )  # Fields to display on the admin list page
    list_filter = ('exchange', )  # Fields that can filter the list page items
    search_fields = ('ticker', 'name')  # Fields that can be searched on the list page
    ordering = ('ticker', )  # Field to order the admin list by


admin.site.register(Stock, StockAdmin)  # Register the model and its admin class
