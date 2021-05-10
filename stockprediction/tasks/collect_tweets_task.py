from django.conf import settings
from django.db import IntegrityError
import tweepy
from stockprediction.models import Tweet, Stock


def collect_tweets(ticker):
    """ Method to collect Stock data from AlphaVantage

    Method collects the Tweets specified for the specified ticker
    from Tweepy and saves them.
    """

    # Authenticate Tweepy credentials
    auth = tweepy.OAuthHandler(settings.TWITTER_CONSUMER_KEY,  settings.TWITTER_SECRET_CONSUMER_KEY)
    auth.set_access_token(settings.TWITTER_TOKEN_KEY, settings.TWITTER_SECRET_TOKEN_KEY)
    api = tweepy.API(auth)

    stock = Stock.objects.get(ticker=ticker)

    # Search for recent Tweets with the specific ticker
    collected_tweets = api.search(q=ticker, result_type='recent', count=100)

    # Iterate over the  collected Tweets and save them
    for tweet in collected_tweets:
        try:
            Tweet.objects.create(
                text=tweet.text,
                created_at=tweet.created_at,
                user_id=tweet.user.id,
                user_screen_name=tweet.user.screen_name,
                verified=tweet.user.verified,
                followers_count=tweet.user.followers_count,
                friends_count=tweet.user.friends_count,
                favourites_count=tweet.user.favourites_count,
                retweet_count=tweet.retweet_count,
                stock=stock,
            )
        except IntegrityError:
            pass
