# stdlib
from typing import List, Tuple, Sequence, Union, Literal, cast
# thirdpartylib
import numpy as np
import pandas as pd
from keras import Model # pyright: ignore[reportMissingTypeStubs]
from keras.models import Sequential # pyright: ignore[reportMissingTypeStubs]
from keras.callbacks import History # pyright: ignore[reportMissingTypeStubs]
from keras.layers import ( # pyright: ignore[reportMissingTypeStubs]
    LSTM, 
    Dense, 
    Dropout,
)
import matplotlib.pyplot as plt
# projectlib
from pv_inverter_modeling.utils.typing import Address, Verbosity
from pv_inverter_modeling.models.io import load_keras_model
from pv_inverter_modeling.config.env import PDM_SAMPLE_DEVICE
from pv_inverter_modeling.utils.logging import Logger
from pv_inverter_modeling.data.schemas import Column
from pv_inverter_modeling.visualization.timeseries import use_dark_theme
from scripts.survival_analysis import DataPipeline


class PredictiveMaintenance(object):
    """
    End-to-end predictive maintenance workflow for inverter power data.

    This class orchestrates the full lifecycle of a sequence-based
    forecasting and monitoring pipeline, including:

    - Loading and preprocessing inverter AC power time-series data
      via a dedicated Polars-based data pipeline
    - Formatting continuous intraday data into fixed-length input and
      output windows suitable for sequence-to-sequence models
    - Defining, training, loading, and running an LSTM-based forecasting
      model
    - Generating qualitative forecasts and visual diagnostics
    - Monitoring recent predicted power generation trends against
      configurable thresholds

    The workflow is designed for exploratory analysis, model-driven
    forecasting, and heuristic-based monitoring rather than strict
    probabilistic inference. Most methods operate on internally stored
    state initialized at construction time.

    Typical usage
    -------------
    >>> pm = PredictiveMaintenance(path, file, verbosity=1)
    >>> pm.train_model()
    >>> pm.testout(index=[0])
    >>> pm.Monitor_Power(numbers_of_days=30, threshold=0.6)
    
    Notes
    -----
    - Time is discretized at 5-minute resolution (288 steps per day).
    - Forecasts are multi-step, predicting full future horizons in a
      single forward pass.
    - Several methods produce visualizations intended for human
      inspection rather than automated decision-making.
    """

    def __init__(
            self,
            path: Address,
            file: Address,
            *,
            inverter: str = PDM_SAMPLE_DEVICE,
            verbosity: Verbosity = 0,
            write_log: bool = False,
        ) -> None:
        """
        Initialize the predictive maintenance workflow.

        This constructor sets up all required preprocessing and logging
        components and prepares the time-series data for downstream
        modeling. Specifically, it:

        - Configures plotting defaults for consistent visualization
        - Instantiates the data preprocessing pipeline
        - Initializes structured logging
        - Formats inverter data into model-ready tensors

        Parameters
        ----------
        path : Address
            Base directory containing the input dataset.
        file : Address
            Filename (or relative path) of the dataset to load.
        inverter : str, default=``PDM_SAMPLE_DEVICE``
            Inverter used for data filtering and subsequent training and
            evaluation.
        verbosity : Verbosity, default=0
            Logging verbosity level.
        write_log : bool, default=False
            Whether to persist logs to disk in addition to stdout.
        """
        # Apply global plotting style for all downstream visualizations
        use_dark_theme()
        # Initialize the data preprocessing pipeline
        self.pipeline = DataPipeline(path, file)
        # Set up structured logging
        self.log = Logger(verbose=verbosity, write_log=write_log)
        # Prepare model-ready training and evaluation datasets
        self.format_data(inverter=inverter)

    def format_data(self, inverter: str) -> None:
        """
        Prepare inverter time-series data for sequence-to-sequence 
        modeling.

        This method performs the final transformation from a dense, 
        daily AC power array into sliding input/output windows suitable 
        for training sequence models (e.g., LSTMs). Specifically, it:

        - Extracts padded daily AC power data for a single inverter
        - Flattens the daily structure into a continuous 1D time series
        - Builds rolling input/output windows measured in full days
        - Reshapes data into 3D tensors expected by Keras LSTM layers
        - Splits the resulting samples into training and test sets

        All resulting arrays and dimensional metadata are stored as
        instance attributes for downstream model training and 
        evaluation.

        Parameters
        ----------
        inverter : str
            Inverter used for data filtering and subsequent training and
            evaluation.
        """
        # Retrieve padded daily AC power data for the target inverter
        data = self.pipeline.select_inverter(inverter=inverter)
        self.log(f"data shape {data.shape}", 1)
        # Flatten daily structure into a single continuous time series
        data_seq = data.flatten()
        # Define input/output horizons in whole days
        input_days = 7
        output_days = 2
        self.input_len = input_days * 288
        self.output_len = output_days * 288
        # Construct sliding input/output windows over the sequence
        X: List[np.ndarray] = []
        y: List[np.ndarray] = []
        iter_range = range(
            len(data_seq) 
            - self.input_len 
            - self.output_len 
            + 1
        )
        for i in iter_range:
            X.append(data_seq[i : i + self.input_len])
            y.append(
                data_seq[
                    i + self.input_len :
                    i + self.input_len + self.output_len
                ]
            )
        # Stack samples into NumPy arrays
        X_np = np.array(X)
        y_np = np.array(y)
        # Reshape into (samples, timesteps, features) for LSTM 
        # compatibility
        X_np = X_np.reshape(X_np.shape[0], self.input_len, 1)
        y_np = y_np.reshape(y_np.shape[0], self.output_len, 1)
        self.log(f"X shape: {X_np.shape}", 1)
        self.log(f"y shape: {y_np.shape}", 1)
        # Split samples into training and test sets
        train_ratio = 0.8
        num_samples = X_np.shape[0]
        self.train_size = int(num_samples * train_ratio)
        self.X_train = X_np[:self.train_size]
        self.y_train = y_np[:self.train_size]
        self.X_test = X_np[self.train_size:]
        self.y_test = y_np[self.train_size:]
        self.log(f"X_train shape: {self.X_train.shape}", 1)
        self.log(f"y_train shape: {self.y_train.shape}", 1)
        self.log(f"X_test shape: {self.X_test.shape}", 1)
        self.log(f"y_test shape: {self.y_test.shape}", 1)
        # Flatten target tensors for loss computation where required
        self.y_train_reshaped = self.y_train.reshape(
            self.y_train.shape[0], self.y_train.shape[1]
        )
        self.y_test_reshaped = self.y_test.reshape(
            self.y_test.shape[0], self.y_test.shape[1]
        )
        self.log(f"y_train_reshape {self.y_train_reshaped.shape}", 1)
        self.log(f"y_test_reshape {self.y_test_reshaped.shape}", 1)

    def model_factory(self) -> Model:
        """
        Construct and compile the LSTM forecasting model.

        This factory method defines a stacked LSTM architecture for
        sequence-to-sequence regression over fixed-length intraday
        windows. The model consumes a univariate time series and 
        predicts a multi-step future horizon in a single forward pass.

        Architecture overview:
        - Two stacked LSTM layers to capture temporal dependencies
        - Dropout regularization to mitigate overfitting
        - Dense projection layers to map latent features to the forecast
        horizon

        The model is compiled with mean squared error loss and mean
        absolute error as an auxiliary metric.

        Returns
        -------
        keras.Model
            A compiled Keras model ready for training or inference.
        """
        model = Sequential([
            # First LSTM layer processes the full input sequence
            LSTM(
                128,
                return_sequences=True,
                input_shape=(2016, 1),
            ),
            Dropout(0.2),
            # Second LSTM summarizes the sequence into a fixed-length vector
            LSTM(
                64,
                return_sequences=False,
            ),
            Dropout(0.2),
            # Fully connected layers map latent state to forecast horizon
            Dense(256, activation="relu"),
            Dense(self.output_len),  # Multi-step forecast output
        ])
        # Compile the model for regression
        model.compile(  # pyright: ignore[reportUnknownMemberType]
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )
        # Emit a summary for verification and debugging
        model.summary()  # pyright: ignore[reportUnknownMemberType]
        return model

    def train_model(self) -> Tuple[Sequential, History]:
        """
        Train the LSTM forecasting model on prepared time-series data.

        This method:
        - Instantiates a fresh model using the configured factory
        - Trains the model on the preformatted training dataset
        - Uses a validation split to monitor generalization during 
          training

        The trained model and its training history are returned for
        downstream evaluation, visualization, or persistence.

        Returns
        -------
        model : keras.Sequential
            Trained Keras model.
        history : keras.callbacks.History
            Training history containing per-epoch loss and metric 
            values.
        """
        # Build a new model instance
        self.model = self.model_factory()
        # Train the model and capture training history
        history = cast(
            History,
            self.model.fit(  # pyright: ignore[reportUnknownMemberType]
                self.X_train,
                self.y_train_reshaped,
                epochs=50,
                batch_size=16,
                validation_split=0.1,
                verbose=1, # pyright: ignore[reportArgumentType]
            ),
        )
        
        return self.model, history
  
    def load_model(self, model_path: Address) -> None:
        """
        Load a previously saved LSTM model from disk.

        This method reconstructs the model architecture using the 
        configured model factory and restores the learned weights from 
        the specified path. The loaded model is assigned to the instance 
        for downstream inference or continued training.

        Parameters
        ----------
        model_path : Address
            Path to the saved model weights.
        """
        # Initialize model architecture and load persisted weights
        self.model = load_keras_model(
            model_path=model_path,
            model_factory=self.model_factory,
            load_from_weights=True,
        )

    def add_event_local_time(self, data: np.ndarray) -> pd.DataFrame:
        """
        Attach a synthetic local-time index to a fixed-length prediction
        array.

        This helper converts a flat prediction array into a time-indexed
        pandas DataFrame by constructing a contiguous intraday timestamp
        range starting at the current day (00:00). It is primarily 
        intended for visualization, debugging, and post-hoc analysis of 
        model outputs where absolute timestamps are not otherwise 
        available.

        The method assumes:
        - 5-minute sampling resolution
        - Exactly two days of data (576 time steps)

        Parameters
        ----------
        data : np.ndarray
            Flat array of prediction values with length 576.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by a synthetic local-time timestamp with a
            single ``value`` column.
        """
        # Anchor timestamps at today's date, normalized to midnight
        start = pd.Timestamp.today().normalize()
        # Construct a fixed 2-day time index at 5-minute resolution
        time_index = pd.date_range(
            start=start,
            periods=576,  # 2 days × 288 steps/day
            freq="5min",
        )
        self.log(f"Length: {len(time_index)}", 1)  # Expected: 576
        # Wrap predictions in a DataFrame with a time index
        df = pd.DataFrame(
            {"value": data},
            index=time_index,
        )
        df.index.name = Column.TIMESTAMP
        # Emit full DataFrame at high verbosity for inspection
        self.log(f"{df}", 2)

        return df

    def testout(self, index: Union[Sequence[int], int]) -> pd.DataFrame:
        """
        Generate and visualize a model forecast for a selected test 
        sample.

        This method runs inference on a specific test window, compares 
        the predicted output against the true future values, and 
        visualizes both as normalized percentage profiles. It is 
        intended for qualitative inspection and debugging rather than 
        formal evaluation.

        Specifically, it:
        - Extracts a single test input/output pair
        - Runs the trained model to generate a multi-step forecast
        - Normalizes past input, true output, and predicted output by
        their respective maxima
        - Plots percentage-based comparisons and summary statistics
        - Returns the predicted output indexed by synthetic local time

        Parameters
        ----------
        index : Sequence[int]
            Index (or indices) into the test set identifying which 
            sample to visualize.

        Returns
        -------
        pandas.DataFrame
            Time-indexed DataFrame containing the predicted values for 
            the selected forecast horizon.
        """
        # Prepare a single test input sample for model inference
        x_input = self.X_test[index].reshape(
            1,
            self.X_test.shape[1],
            1,
        )
        # Extract corresponding ground-truth future values
        y_true = self.y_test[index].reshape(
            self.y_test.shape[1]
        )
        # Run model inference to obtain forecast
        y_pred = cast(
            np.ndarray,
            self.model.predict(  # pyright: ignore[reportUnknownMemberType]
                x_input
            ),
        )
        y_pred = y_pred.reshape(self.y_test.shape[1])
        # Normalize input, true output, and predictions as percentages
        true_max: float = y_true.max()
        pred_max: float = y_pred.max()
        input_max: float = x_input.max()
        y_true_pct = (y_true / true_max) * 100
        y_pred_pct = (y_pred / pred_max) * 100
        x_input_pct = (x_input.flatten() / input_max) * 100
        # Visualize percentage-based comparison of true vs predicted 
        # values
        plt.figure(figsize=(14, 6))  # pyright: ignore[reportUnknownMemberType]
        plt.plot(  # pyright: ignore[reportUnknownMemberType]
            y_true_pct,
            label="Actual 2 Days (%)",
        )
        plt.plot(  # pyright: ignore[reportUnknownMemberType]
            y_pred_pct,
            label="Predicted 2 Days (%)",
        )
        # Overlay average reference lines
        plt.axhline(  # pyright: ignore[reportUnknownMemberType]
            y=float(np.mean(y_pred_pct)),
            color="r",
            linestyle="--",
            label="Predicted avg (%)",
        )
        plt.axhline(  # pyright: ignore[reportUnknownMemberType]
            y=float(np.mean(y_true_pct)),
            color="g",
            linestyle="--",
            label="True avg (%)",
        )
        plt.axhline(  # pyright: ignore[reportUnknownMemberType]
            y=float(np.mean(x_input_pct)),
            color="y",
            linestyle="--",
            label="Past 7-day avg (%)",
        )
        plt.title(  # pyright: ignore[reportUnknownMemberType]
            "Predicted Power Generated for the Next 2 Days"
        )
        plt.xlabel(  # pyright: ignore[reportUnknownMemberType]
            "Index points for 2 days"
        )
        plt.ylabel(  # pyright: ignore[reportUnknownMemberType]
            "Percentage (%)"
        )
        plt.legend()  # pyright: ignore[reportUnknownMemberType]
        plt.grid(True)  # pyright: ignore[reportUnknownMemberType]
        plt.show()  # pyright: ignore[reportUnknownMemberType]

        # Attach a synthetic local-time index to the prediction for 
        # inspection
        output = self.add_event_local_time(y_pred)

        return output

    def detect_power_gen(self, number_of_days: int) -> List[float]:
        """
        Estimate average predicted power generation over recent test 
        samples.

        This method runs model inference on the last ``number_of_days`` 
        test windows and computes the mean predicted value for each 
        forecast horizon. The resulting sequence provides a coarse, 
        scalar summary of expected power generation trends over the 
        selected period.

        It is primarily intended for:
        - Trend detection
        - Comparative analysis across recent days
        - Downstream heuristics rather than precise forecasting

        Parameters
        ----------
        number_of_days : int
            Number of most recent test samples to evaluate.

        Returns
        -------
        list[float]
            List of mean predicted values, one per evaluated test 
            sample.
        """
        result: List[float] = []
        # Iterate over the last `number_of_days` samples in the test set
        iter_range = range(
            self.X_test.shape[0] - number_of_days,
            self.X_test.shape[0],
        )
        for i in iter_range:
            # Prepare a single test input for inference
            x_input = self.X_test[i].reshape(
                1,
                self.X_test.shape[1],
                1,
            )
            # Run model prediction for the selected window
            y_pred = cast(
                np.ndarray,
                self.model.predict(  # pyright: ignore[reportUnknownMemberType]
                    x_input,
                    verbose=0, # pyright: ignore[reportArgumentType]
                ),
            )
            y_pred = y_pred.reshape(self.y_test.shape[1])
            # Reduce the multi-step forecast to a scalar summary 
            # statistic
            res = float(np.mean(y_pred))
            result.append(res)

        return result
    
    def monitor_power(
            self,
            numbers_of_days: int,
            threshold: float,
            *,
            threshold_mode: Literal["abs", "pct"] = "pct",
        ) -> None:
        """
        Visualize recent predicted power generation and flag 
        low-performance trends.

        This method aggregates model-based power generation estimates 
        over the most recent test windows and visualizes them as 
        percentages of a reference maximum. A horizontal threshold line 
        is overlaid to mark a minimum acceptable performance level.

        The threshold can be interpreted either as:
        - an absolute power value ("absolute"), or
        - a percentage of the observed maximum ("percentage").

        Parameters
        ----------
        numbers_of_days : int
            Number of most recent test samples to include.
        threshold : float
            Threshold value, interpreted according to 
            ``threshold_mode``.
        threshold_mode : {"abs", "pct"}, default "pct"
            Interpretation of ``threshold``.
        """
        # Compute mean predicted power generation for recent test 
        # samples
        result_out = self.detect_power_gen(numbers_of_days)
        # Normalize values as a percentage of the maximum observed value
        max_val = max(result_out)
        if max_val <= 0:
            raise ValueError("Maximum predicted power must be positive.")
        result_pct = [(x / max_val) * 100 for x in result_out]
        # Resolve threshold into percentage space
        if threshold_mode == "abs":
            threshold_pct = (threshold / max_val) * 100
            threshold_label = f"Threshold ({threshold:.2f} abs)"
        else:
            threshold_pct = threshold
            threshold_label = f"Threshold ({threshold:.1f}%)"
        # Set up scatter plot for percentage-based monitoring
        plt.figure(figsize=(14, 6))  # pyright: ignore[reportUnknownMemberType]
        x_coordinates = list(range(len(result_out)))
        plt.scatter(  # pyright: ignore[reportUnknownMemberType]
            x_coordinates,
            result_pct,
        )
        plt.xlabel(  # pyright: ignore[reportUnknownMemberType]
            "Index of Recent Test Samples"
        )
        plt.ylabel(  # pyright: ignore[reportUnknownMemberType]
            "Mean Power Generated (%)"
        )
        plt.title(  # pyright: ignore[reportUnknownMemberType]
            "Predicted Power Generation Over Recent Days"
        )
        # Overlay threshold line indicating alert condition
        plt.axhline(  # pyright: ignore[reportUnknownMemberType]
            y=threshold_pct,
            color="r",
            linestyle="--",
            label=threshold_label,
        )
        # Annotate threshold region for clarity
        plt.text(  # pyright: ignore[reportUnknownMemberType]
            x=5,
            y=threshold_pct,
            s="SENDING NOTICE",
            fontsize=12,
            color="blue",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
                pad=2,
            ),
        )
        plt.legend()  # pyright: ignore[reportUnknownMemberType]
        plt.grid(True)  # pyright: ignore[reportUnknownMemberType]
        plt.show()  # pyright: ignore[reportUnknownMemberType]
