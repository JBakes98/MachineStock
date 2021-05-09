from django.contrib import admin
from stockprediction.models import Tweet


class TweetAdmin(admin.ModelAdmin):
    fieldsets = (
        (None, {'fields': (
            'stock',
            'text',
            'created_at',
            'user_screen_name',
            'favourites-count',
            'retweet_count',
        )}),
    )

    list_display = ('stock', 'text', 'created_at', 'favourites_count', 'retweet_count')
    list_filter = ('stock', 'user_screen_name', 'created_at')
    search_fields = ('stock', 'user_screen_name', 'created_at')
    ordering = ('created_at', 'stock')


admin.site.register(Tweet, TweetAdmin)
