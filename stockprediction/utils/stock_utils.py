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
