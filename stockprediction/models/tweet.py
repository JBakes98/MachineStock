from django.db import models
from .stock import Stock


class Tweet(models.Model):
    text = models.CharField(max_length=280, blank=False, null=False)
    created_at = models.DateTimeField(blank=False, null=False)
    user_id = models.BigIntegerField()
    user_screen_name = models.CharField(max_length=50, null=False, blank=False)
    verified = models.BooleanField(default=False)
    followers_count = models.BigIntegerField()
    friends_count = models.BigIntegerField()
    favourites_count = models.BigIntegerField()
    retweet_count = models.BigIntegerField()
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, blank=False, null=False)

    class Meta:
        verbose_name_plural = 'Tweets'

        ordering = ['-created_at']

        indexes = [
            models.Index(fields=['stock', 'text'])
        ]

        constraints = [
            models.UniqueConstraint(fields=['text', 'created_at'], name='tweet_data')
        ]

    def __str__(self):
        return self.text
