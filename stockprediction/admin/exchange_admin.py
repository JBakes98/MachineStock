from django.contrib import admin

from stockprediction.models import Exchange


class ExchangeAdmin(admin.ModelAdmin):
    """ Class that defines the Exchange models admin properties """

    # Fields to display on the admin detail page
    fieldsets = (
        (None, {'fields': (
            'symbol',
            'name',
        )}),
    )

    list_display = ('symbol', 'name')  # Fields to display on the admin list page
    search_fields = ('symbol', 'name')  # Fields that can be searched on the list page
    ordering = ('symbol',)  # Field to order the admin list by


admin.site.register(Exchange, ExchangeAdmin)  # Register the model and its admin class
