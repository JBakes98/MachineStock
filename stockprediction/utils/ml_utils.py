from typing import Any, Dict, List
from pytorch_forecasting.metrics import (
    Metric,
    MultiLoss,
    QuantileLoss,
)


def to_list(value: Any) -> List[Any]:
    """Convert value or list to list of values

    Parameters
    ----------
    value : Any
        Value to convert
    """

    # If its already a list, return object
    if isinstance(value, (tuple, list)) and not isinstance(value, rnn.PackedSequence):
        return value
    else:
        return [value]


def to_prediction(out: Dict[str, Any]):
    """ Convert output to prediction using the loss metric.

    Parameters
    ----------
    out : (Dict[str, Any])
        output of network where "prediction" has been transformed
    """

    # If samples are already drawn, directly take the mean
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
    """ Convert output to quantiles using the loss metric.

    Parameters
    ----------
    out : (Dict[str, Any])
        Output of network where "prediction" has been transformed
    **kwargs : Dict
        arguments for metric 'to_quantiles' method
    """

    # If samples are directly output, take quantiles
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
