import numpy as np
import pandas as pd


def format_stock_dataset_for_db(dataset: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """ Format Dataframe prepared for saving to StockData model

        Parameters
        ----------
        dataset : pandas.Dataframe
            A dataframe that need to be formatted
        ticker: str
            The ticker of the stock the data is for
    """
    # Format DataFrame by moving the date index into a column,
    # flip the data so most recent data is at the bottom for when rolling
    # averages are calculated, drop the unnecessary data and simplify remaining
    # column names.
    dataset.reset_index(inplace=True)
    dataset = dataset.sort_index(ascending=False)
    dataset.drop(['8. split coefficient'], axis=1, inplace=True)
    dataset = dataset.rename(columns={'1. open': 'open',
                                      '2. high': 'high',
                                      '3. low': 'low',
                                      '4. close': 'close',
                                      '5. adjusted close': 'adj close',
                                      '6. volume': 'volume',
                                      '7. dividend amount': 'dividend amount',
                                      })
    # Add stock column to identify what stock the data relates to
    dataset['stock'] = ticker

    return dataset
