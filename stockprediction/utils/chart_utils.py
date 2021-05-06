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


def to_list(value: Any) -> List[Any]:
    """
    Convert value or list to list of values.
    If already list, return object directly

    Args:
        value (Any): value to convert

    Returns:
        List[Any]: list of values
    """
    if isinstance(value, (tuple, list)) and not isinstance(value, rnn.PackedSequence):
        return value
    else:
        return [value]


def to_prediction(out: Dict[str, Any], **kwargs):
    """
        Convert output to prediction using the loss metric.

        Args:
            out (Dict[str, Any]): output of network where "prediction" has been
                transformed with :py:meth:`~transform_output`
            **kwargs: arguments to metric ``to_quantiles`` method

        Returns:
            torch.Tensor: predictions of shape batch_size x timesteps
        """
    # if samples were already drawn directly take mean
    if out.get("output_transformation", True) is None:
        if isinstance(QuantileLoss(), MultiLoss):
            out = [Metric.to_prediction(loss, out["prediction"][idx]) for idx, loss in enumerate(QuantileLoss())]
        else:
            out = Metric.to_prediction(QuantileLoss(), out["prediction"])
    else:
        try:
            out = QuantileLoss().to_prediction(out["prediction"])
        except TypeError:  # in case passed kwargs do not exist
            out = QuantileLoss().to_prediction(out["prediction"])
    return out


def to_quantiles(out: Dict[str, Any], **kwargs):
    """
        Convert output to quantiles using the loss metric.

        Args:
            out (Dict[str, Any]): output of network where "prediction" has been
                transformed with :py:meth:`~transform_output`
            **kwargs: arguments to metric ``to_quantiles`` method

        Returns:
            torch.Tensor: quantiles of shape batch_size x timesteps x n_quantiles
        """
    # if samples are output directly take quantiles
    if out.get("output_transformation", True) is None:
        if isinstance(QuantileLoss().loss, MultiLoss):
            out = [
                Metric.to_quantiles(loss, out["prediction"][idx], quantiles=kwargs.get("quantiles", loss.quantiles))
                for idx, loss in enumerate(QuantileLoss())
            ]
        else:
            out = Metric.to_quantiles(
                QuantileLoss(), out["prediction"], quantiles=kwargs.get("quantiles", QuantileLoss().quantiles)
            )
    else:
        try:
            out = QuantileLoss().to_quantiles(out["prediction"])
        except TypeError:  # in case passed kwargs do not exist
            out = QuantileLoss().to_quantiles(out["prediction"])
    return out


def plot_prediction(x: Dict[str, torch.Tensor],
                    out: Dict[str, torch.Tensor],
                    idx: int = 0,
                    show_future_observed: bool = True):
    encoder_targets = to_list(x['encoder_target'])
    decoder_targets = to_list(x['decoder_target'])

    prediction_kwargs = {}
    quantiles_kwargs = {}

    y_raws = to_list(out['prediction'])
    y_hats = to_list(to_prediction(out, **prediction_kwargs))
    y_quantiles = to_list(to_quantiles(out, **quantiles_kwargs))

    figs = []
    for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
            y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
    ):
        y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
        max_encoder_length = x['encoder_lengths'].max()
        y = torch.cat(
            (
                y_all[: x['encoder_lengths'][idx]],
                y_all[max_encoder_length: (max_encoder_length + x['decoder_lengths'][idx])],
            ),
        )
        print(dir(y_hat))
        y_hat = y_hat.detach().cpu()[idx, : x['decoder_lengths'][idx]]
        y_quantile = y_quantile.detach().cpu()[idx, : x['decoder_lengths'][idx]]
        y_raw = y_raw.detach().cpu()[idx, : x['decoder_lengths'][idx]]

        y = y.detach().cpu()

        n_pred = y_hat.shape[0]
        x_obs = np.arange(-(y.shape[0] - n_pred), 0)
        x_pred = np.arange(n_pred)

        # Plot observed history
        if len(x_obs) > 0:
            layout = {
                "title": {"text": "Machine Learning Prediction"},
                "xaxis": {"title": "Time Index"},
                "yaxis": {"title": "Adjusted Close ($)"},
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

            trace1 = {
                'line': {'width': 1},
                'name': 'Observed',
                'type': 'scatter',
                'x': x_obs,
                'y': y[:-n_pred:],
                'legendgroup': 'Observed',
                'marker': {'color': '#000000'},
            }
            trace2 = {
                'line': {'width': 1},
                'name': 'Observed Future',
                'type': 'scatter',
                'x': x_pred,
                'y': y[-n_pred:],
                'legendgroup': 'Observed Prediction',
                'marker': {'color': '#000000'},
            }
            trace3 = {
                'line': {'width': 1},
                'name': 'Prediction',
                'type': 'scatter',
                'x': x_pred,
                'y': y_hat,
                'legendgroup': 'Prediction',
                'marker': {'color': '#FF0000'},
            }
            trace4 = {
                'line': {'width': 1},
                'name': 'Predicted Quantiles',
                'type': 'scatter',
                'x': x_pred,
                'y': y_quantile[:, y_quantile.shape[1] // 2],
                'legendgroup': 'Predicted Quantiles',
                'marker': {'color': '#FFA500'},
            }

            if show_future_observed:
                data = ([trace1, trace2, trace3, trace4])
            else:
                data = ([trace1, trace3, trace4])

            plot_div = plot(Figure(data=data, layout=layout), output_type='div')
            figs.append(plot_div)

    if isinstance(x['encoder_target'], (tuple, list)):
        return figs
    else:
        return plot_div
