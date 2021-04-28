from django.contrib import admin

from stockprediction.models import Exchange


class ExchangeAdmin(admin.ModelAdmin):
    fieldsets = (
        (None, {'fields': (
            'symbol',
            'name',
        )}),
    )

    list_display = ('symbol', 'name')
    search_fields = ('symbol', 'name')
    ordering = ('symbol',)


admin.site.register(Exchange, ExchangeAdmin)
