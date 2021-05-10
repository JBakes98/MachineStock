from django.db import models
from .stock import Stock


class Tweet(models.Model):
    """
    A class used to represent an Exchange

    ...

    Attributes
    ----------
    text : CharField()
        The text of the Tweet
    created_at : DateTimeField()
        The date and time the Tweet was posted on Twitter
    user_id : BigIntegerField()
        The Twitter user id of the Tweet author
    user_screen_name : CharField()
        The Twitter users username of the Tweet author
    verified : BooleanField()
        If the Tweet is verified
    followers_count : BigIntegerField()
        The number of followers the user has
    favourites_count : BigIntegerField()
        The number of favourites the Tweet has
    retweet_count : BigIntegerField()
        The number of retweets the Tweet has
    stock : ForeignKey()
        The Stock the Tweet relates to

    Methods
    -------
    __str__()
        Returns the text attribute of the Tweet
    """

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
