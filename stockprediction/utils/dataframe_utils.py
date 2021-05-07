import numpy as np
import pandas as pd


def format_stock_dataset_for_db(dataset: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """ Format DataFrame prepared for saving to StockData model

        Parameters
        ----------
        dataset : pandas.DataFrame
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


def format_stock_dataset_for_ml(dataset: pd.DataFrame):
    dataset.fillna(value=0, inplace=True)

    dataset['ticker'] = dataset['ticker'].astype('category')
    dataset['exchange'] = dataset['exchange'].astype('category')
    dataset['day'] = dataset['date'].dt.day_name()
    dataset['day'] = dataset['day'].astype('category')
    dataset['month'] = dataset['date'].dt.strftime('%B')
    dataset['month'] = dataset['month'].astype('category')

    # When data is retrieved
    dataset['open'] = pd.to_numeric(dataset['open'], errors='coerce')
    dataset['close'] = pd.to_numeric(dataset['close'], errors='coerce')
    dataset['low'] = pd.to_numeric(dataset['low'], errors='coerce')
    dataset['high'] = pd.to_numeric(dataset['high'], errors='coerce')
    dataset['adj_close'] = pd.to_numeric(dataset['adj_close'], errors='coerce')
    dataset['dividend_amount'] = pd.to_numeric(dataset['dividend_amount'], errors='coerce')
    dataset['change'] = pd.to_numeric(dataset['change'], errors='coerce')
    dataset['change_perc'] = pd.to_numeric(dataset['change_perc'], errors='coerce')
    dataset['ma7'] = pd.to_numeric(dataset['ma7'], errors='coerce')
    dataset['ma21'] = pd.to_numeric(dataset['ma21'], errors='coerce')
    dataset['ema12'] = pd.to_numeric(dataset['ema12'], errors='coerce')
    dataset['ema26'] = pd.to_numeric(dataset['ema26'], errors='coerce')
    dataset['MACD'] = pd.to_numeric(dataset['MACD'], errors='coerce')
    dataset['sd20'] = pd.to_numeric(dataset['sd20'], errors='coerce')
    dataset['upper_band'] = pd.to_numeric(dataset['upper_band'], errors='coerce')
    dataset['lower_band'] = pd.to_numeric(dataset['lower_band'], errors='coerce')
    dataset['ema'] = pd.to_numeric(dataset['ema'], errors='coerce')
    dataset['momentum'] = pd.to_numeric(dataset['momentum'], errors='coerce')
    dataset['momentum_log'] = pd.to_numeric(dataset['log_momentum'], errors='coerce')

    # Group by ticker and sort by date and increment by 1 to
    # denote order of time series
    dataset['time_idx'] = dataset.sort_values(['date'], ascending=True).groupby(['ticker']).cumcount() + 1

    return dataset
