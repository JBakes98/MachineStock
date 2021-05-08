from plotly.offline import plot
from plotly.graph_objects import Figure

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import rnn
from pytorch_forecasting.metrics import (
    Metric,
    MultiLoss,
    QuantileLoss,
)


def plot_tech_indicators(dataset: pd.DataFrame, stock: str):
    # Replace 0 with Nan so indicators such as ma that don't have a value
    # until 7 days of data don't display inaccurate data
    dataset.replace(0, np.nan, inplace=True)

    trace1 = {
        'name': stock,
        'type': 'candlestick',
        'x': dataset['date'],
        'yaxis': 'y2',
        'low': dataset['low'],
        'high': dataset['high'],
        'open': dataset['open'],
        'close': dataset['close'],
    }
    trace2 = {
        "line": {"width": 1},
        "mode": "lines",
        "name": "Moving Average",
        "type": "scatter",
        "x": dataset['date'],
        "y": dataset['ma7'],
        "yaxis": "y2",
    }
    trace3 = {
        "name": "Volume",
        "type": "bar",
        "x": dataset['date'],
        "y": dataset['volume'],
        "yaxis": "y",

    }
    trace4 = {
        "line": {"width": 1},
        "name": "Bollinger Bands",
        "type": "scatter",
        "x": dataset['date'],
        "y": dataset['upper_band'],
        "yaxis": "y2",
        "marker": {"color": "#ccc"},
        "hoverinfo": "none",
        "legendgroup": "Bollinger Bands"
    }
    trace5 = {
        "line": {"width": 1},
        "type": "scatter",
        "x": dataset['date'],
        "y": dataset['lower_band'],
        "yaxis": "y2",
        "marker": {"color": "#ccc"},
        "hoverinfo": "none",
        "showlegend": False,
        "legendgroup": "Bollinger Bands"
    }
    data = ([trace1, trace2, trace3, trace4, trace5])

    layout = {
        "xaxis": {"rangeselector": {
            "x": 0,
            "y": 0.9,
            "font": {"size": 13},

            "buttons": [
                {
                    "step": "all",
                    "count": 1,
                    "label": "reset"
                },
                {
                    "step": "month",
                    "count": 3,
                    "label": "3 mo",
                    "stepmode": "backward"
                },
                {
                    "step": "month",
                    "count": 1,
                    "label": "1 mo",
                    "stepmode": "backward"
                },
                {"step": "all"}
            ]
        }},
        "yaxis": {
            "domain": [0, 0.2],
            "showticklabels": False,
        },
        "legend": {
            "x": 0.3,
            "y": 0.9,
            "yanchor": "bottom",
            "orientation": "h"
        },
        "margin": {
            "b": 30,
            "l": 30,
            "r": 30,
            "t": 30,
        },
        "yaxis2": {"domain": [0.2, 0.8]},
        "plot_bgcolor": "rgb(250, 250, 250)"
    }

    plot_div = plot(Figure(data=data, layout=layout), output_type='div')

    return plot_div

