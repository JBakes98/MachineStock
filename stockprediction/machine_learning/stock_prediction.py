import os
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import Baseline, SMAPE, TemporalFusionTransformer, QuantileLoss, GroupNormalizer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from stock_prediction.charting import plot_prediction


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
        self.ticker = None
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

    def create_time_series(self):
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

    def _create_validation_dataset(self):
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training,
            self.dataset,
            predict=True,
            stop_randomization=True
        )

    def _create_dataloaders(self):
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

    def train_model(self):
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

    def load_model(self):
        if self.training is None:
            self.create_time_series()
        if self.validation is None:
            self._create_validation_dataset()
        if self.val_dataloader is None:
            self._create_dataloaders()

        module_dir = os.path.dirname(__file__)  # get current directory
        file_path = os.path.join(module_dir, 'models/tft-model')

        # If no saved model is found, train the model
        if not file_path.exists():
            self.train_model()

        # Load the model from the save
        tft = torch.load(file_path)
        self.model = tft

    def plot_test_predictions(self):
        # If learning model is not present load it
        if self.model is None:
            self.load_model()

        raw_predictions, x = self.model.predict(
            self.val_dataloader,
            mode="raw",
            return_x=True
        )

        plot = plot_prediction(x, raw_predictions, idx=0)

        return plot

    def plot_future_predictions(self):
        # Load model if not assigned
        if self.model is None:
            self.load_model()

        encoder_data = self.dataset[lambda x: x.time_idx > x.time_idx.max() - self.max_encoder_length]
        last_data = self.dataset[lambda x: x.time_idx == x.time_idx.max()]

        decoder_data = pd.concat(
            [last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in
             range(1, self.max_prediction_length + 1)],
            ignore_index=True
        )

        decoder_data['time_idx'] = decoder_data['date'].dt.year * 12 + decoder_data['date'].dt.month
        decoder_data['time_idx'] += encoder_data['time_idx'].max() + 1 - decoder_data['time_idx'].min()

        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        new_raw_predictions, new_x = self.model.predict(new_prediction_data, mode='raw', return_x=True)

        plot = plot_prediction(new_x, new_raw_predictions, idx=0, show_future_observed=False)

        return plot

