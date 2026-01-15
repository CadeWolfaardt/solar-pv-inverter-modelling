# thirdpartylib
from torch import Tensor
from torch.nn import Module, LSTM, Linear


class LSTMRegressor(Module):
    """
    LSTM-based regressor for sequence-to-one prediction tasks.

    This model processes an input sequence using an LSTM layer and
    produces a single scalar prediction based on the final time step's
    hidden state. It is commonly used for time-series forecasting where
    each input sequence maps to a single continuous target value.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Number of hidden units in the LSTM layer.
    """
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        # Recurrent layer that processes the input sequence
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        # Linear projection from final hidden state to scalar output
        self.fc = Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the LSTM regressor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape 
            ``(batch_size, sequence_length, input_size)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, 1)``, representing the
            predicted scalar value for each input sequence.
        """
        # LSTM outputs hidden states for all time steps
        out, _ = self.lstm(x)
        # Extract hidden state from the final time step
        last = out[:, -1, :]
        # Map final hidden state to regression output
        return self.fc(last)
