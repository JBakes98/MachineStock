from django.contrib import admin
from stockprediction.models import Tweet


class TweetAdmin(admin.ModelAdmin):
    """ Class that defines the Tweet models admin properties """

    # Fields to display on the admin detail page
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

    # Fields to display on the admin list page
    list_display = ('stock', 'text', 'created_at', 'favourites_count', 'retweet_count')
    list_filter = ('stock', 'user_screen_name', 'created_at')  # Fields that can filter the list page items
    search_fields = ('stock', 'user_screen_name', 'created_at') # Fields that can be searched on the list page
    ordering = ('created_at', 'stock')  # Field to order the admin list by


admin.site.register(Tweet, TweetAdmin)  # Register the model and its admin class
