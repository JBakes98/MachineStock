import numpy as np
import pandas as pd


def get_technical_indicators(dataset: pd.DataFrame, col: str) -> pd.DataFrame:
    """ Method that calculates technical indicators of a dataset

    Method that calculates the technical indicators of the specified
    column in the provided dataset.

    Parameters
        ----------
        dataset : pandas.DataFrame
            A dataframe that contains the data to calculate the technical indicators

        col : str
            The datasets column that the technical indicators should be calculated for
    """

    # Calculate the change in stock value from previous day in absolute
    # and percentage terms
    dataset['change'] = dataset.close.diff()
    dataset['change_perc'] = dataset.close.pct_change() * 100
    dataset['change'] = dataset.change.fillna(0).astype(float)
    dataset['change_perc'] = dataset.change_perc.fillna(0).astype(float)

    # Calculate 7 and 21 days moving average
    dataset['ma7'] = dataset[col].rolling(window=7).mean()
    dataset['ma21'] = dataset[col].rolling(window=21).mean()

    # Calculate MACD: Provides exponential weighted functions.
    dataset['ema26'] = dataset[col].ewm(span=26).mean()
    dataset['ema12'] = dataset[col].ewm(span=12).mean()
    dataset['MACD'] = (dataset['ema12'] - dataset['ema26'])

    # Calculate Bollinger Bands
    dataset['sd20'] = dataset[col].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + dataset['sd20'] * 2
    dataset['lower_band'] = dataset['ma21'] - dataset['sd20'] * 2

    # Calculate exponential moving average
    dataset['ema'] = dataset[col].ewm(com=0.5).mean()

    # Calculate momentum
    dataset['momentum'] = dataset[col] - 1
    dataset['log_momentum'] = np.log(dataset[col] - 1)

    dataset.fillna(value=0, inplace=True)

    return dataset


def format_stock_dataset_for_ml(dataset: pd.DataFrame):
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
