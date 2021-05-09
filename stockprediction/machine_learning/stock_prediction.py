import os
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas._libs.tslibs.offsets import BDay

import torch
from torch.nn.utils import rnn
import pytorch_lightning as pl
from pytorch_forecasting import Baseline, SMAPE, TemporalFusionTransformer, QuantileLoss, GroupNormalizer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_forecasting.metrics import QuantileLoss

from plotly.offline import plot
from plotly.graph_objects import Figure

from stockprediction.utils import ml_utils


class StockMachineLearning:
    def __init__(self,
                 dataset: pd.DataFrame,
                 ticker: str,
                 max_prediction_length: int = 30,
                 max_encoder_length: int = 70,
                 batch_size: int = 16,
                 epochs=100
                 ):
        self.dataset = dataset
        self.ticker = ticker
        self.ticker = ticker
        self.stock_idx = self._get_stock_idx()
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.training_cutoff = self.dataset['time_idx'].max() - self.max_prediction_length
        self.batch_size = batch_size
        self.epochs = epochs
        # Attributes that are created through class methods
        self.training = None
        self.validation = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.model = None  # Learning model that can be created or loaded

    def create_time_series(self) -> None:
        self.training = TimeSeriesDataSet(
            self.dataset[lambda x: x.time_idx < self.training_cutoff],
            time_idx='time_idx',
            target='adj_close',
            group_ids=['ticker', 'exchange'],
            min_encoder_length=1,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=['ticker', 'exchange'],
            time_varying_known_categoricals=['day', ],
            time_varying_unknown_categoricals=[],
            time_varying_known_reals=['time_idx', ],
            time_varying_unknown_reals=[
                'open',
                'high',
                'low',
                'close',
                'volume',
                'adj_close',
                'change',
                'change_perc',
                'dividend_amount',
                'ma7',
                'ma21',
                'ema12',
                'ema26',
                'momentum',
                'MACD',
            ],
            target_normalizer=GroupNormalizer(
                groups=['ticker'], transformation='softplus'
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

    def _create_validation_dataset(self) -> None:
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training,
            self.dataset,
            predict=True,
            stop_randomization=True
        )

    def _create_dataloaders(self) -> None:
        self.train_dataloader = self.training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=0
        )
        self.val_dataloader = self.validation.to_dataloader(
            train=False,
            batch_size=self.batch_size * 10,
            num_workers=0
        )

    def _get_stock_idx(self) -> int:
        stocks = self.dataset.ticker.unique()
        i = 0

        while i < len(stocks):
            if stocks[i] == self.ticker:
                break
            i += 1

        return i

    def train_model(self) -> None:
        if self.training is None:
            self.create_time_series()
        self._create_validation_dataset()
        self._create_dataloaders()

        actuals = torch.cat([y for x, (y, weight) in iter(self.val_dataloader)])
        baseline_predictions = Baseline().predict(self.val_dataloader)
        (actuals - baseline_predictions).abs().mean().item()

        pl.seed_everything(42)

        # Define the trainer with early stopping
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=False,
            mode='min'
        )
        lr_logger = LearningRateMonitor()

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            gpus=0,
            # Clipping gradients is a hyperparameter and important to prevent divergence of
            # the gradient for recurrent neural networks
            gradient_clip_val=0.1,
            limit_train_batches=50,
            callbacks=[lr_logger, early_stop_callback],
        )

        # Create the model
        tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.3,
            hidden_size=512,
            lstm_layers=6,
            dropout=0.1,
            output_size=7,
            loss=QuantileLoss(),
            attention_head_size=12,
            max_encoder_length=self.max_encoder_length,
            hidden_continuous_size=128,
            log_interval=1,
            # Reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=4,
        )

        trainer.fit(
            tft,
            train_dataloader=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )

        # Load the best model according to the validation loss (given that
        # early stopping is used, this is not necessarily the last epoch)
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        # Save the model to the models directory
        torch.save(best_tft, 'models/tft-model')

    def load_model(self) -> None:
        if self.training is None:
            self.create_time_series()
        if self.validation is None:
            self._create_validation_dataset()
        if self.val_dataloader is None:
            self._create_dataloaders()

        module_dir = os.path.dirname(__file__)  # get current directory
        file_path = os.path.join(module_dir, 'models/tft-model')

        # If no saved model is found, train the model
        if not os.path.exists(file_path):
            self.train_model()

        # Load the model from the save
        tft = torch.load(file_path)
        self.model = tft

    def plot_test_predictions(self) -> Figure:
        # If learning model is not present load it
        if self.model is None:
            self.load_model()

        raw_predictions, x = self.model.predict(
            self.val_dataloader,
            mode="raw",
            return_x=True
        )

        plot = self._plot_prediction(x, raw_predictions, idx=self.stock_idx)

        return plot

    def plot_future_predictions(self) -> Figure:
        # Load model if not assigned
        if self.model is None:
            self.load_model()

        encoder_data = self.dataset[lambda x: x.time_idx > x.time_idx.max() - self.max_encoder_length]
        last_data = self.dataset[lambda x: x.time_idx == x.time_idx.max()]

        decoder_data = pd.concat(
            [last_data.assign(date=lambda x: x.date + 0 * BDay()) for _ in
             range(1, self.max_prediction_length + 1)],
            ignore_index=True
        )

        decoder_data['time_idx'] = decoder_data.sort_values(['date'], ascending=True).groupby(['ticker']).cumcount() + 1
        decoder_data['time_idx'] += encoder_data['time_idx'].max() + 1 - decoder_data['time_idx'].min()

        decoder_data['month'] = decoder_data['date'].dt.strftime('%B')
        decoder_data['month'] = decoder_data['month'].astype('category')
        decoder_data['day'] = decoder_data['date'].dt.day_name()
        decoder_data['day'] = decoder_data['day'].astype('category')

        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        new_raw_predictions, new_x = self.model.predict(new_prediction_data, mode='raw', return_x=True)

        plot = self._plot_prediction(new_x, new_raw_predictions, idx=self.stock_idx, show_future_observed=False)

        return plot

    def _plot_prediction(
            self, x: Dict[str, torch.Tensor],
            out: Dict[str, torch.Tensor],
            idx: int = 0,
            show_future_observed: bool = True
    ) -> Figure:
        encoder_targets = ml_utils.to_list(x['encoder_target'])
        decoder_targets = ml_utils.to_list(x['decoder_target'])

        prediction_kwargs = {}
        quantiles_kwargs = {}

        y_raws = ml_utils.to_list(out['prediction'])
        y_hats = ml_utils.to_list(ml_utils.to_prediction(out, **prediction_kwargs))
        y_quantiles = ml_utils.to_list(ml_utils.to_quantiles(out, **quantiles_kwargs))

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

            y_hat = y_hat.detach().cpu()[idx, : x['decoder_lengths'][idx]]
            y_quantile = y_quantile.detach().cpu()[idx, : x['decoder_lengths'][idx]]
            y_raw = y_raw.detach().cpu()[idx, : x['decoder_lengths'][idx]]

            y = y.detach().cpu()

            n_pred = y_hat.shape[0]
            x_obs = np.arange(-(y.shape[0] - n_pred), 0)
            x_pred = np.arange(n_pred)

            interpretation = self.model.interpret_output(out)
            encoder_length = x['encoder_lengths'][self.stock_idx]
            loss = self.model.loss
            loss_value = loss(y_raw[None], (y[-n_pred:][None], None))

            # Plot observed history
            layout = {
                "title": f"Loss {loss_value}",
                "xaxis": {"title": "Time Index"},
                "yaxis": {"title": "Adjusted Close"},
                "yaxis2": {
                    "title": "Attention",
                    "domain": [0.2, 0.8],
                    "overlaying": "y",
                    "side": "right",
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
            trace5 = {
                'line': {'width': 1},
                'name': 'Attention',
                'type': 'scatter',
                'x': torch.arange(-encoder_length, 0),
                'y': interpretation['attention'][self.stock_idx, :encoder_length].detach().cpu(),
                'legendgroup': 'Attention',
                'marker': {'color': '#aca4e0'},
                'yaxis': 'y2',
            }

            if show_future_observed:
                data = ([trace1, trace2, trace3, trace4, trace5])
            else:
                data = ([trace1, trace3, trace4, trace5])

            plot_div = plot(Figure(data=data, layout=layout), output_type='div')

            return plot_div
