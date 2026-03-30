"""
lstm_model.py
PyTorch LSTM for gentrification risk scoring.
"""

import torch
import torch.nn as nn


class GentrificationLSTM(nn.Module):
    """
    LSTM that takes a sequence of neighborhood features and outputs
    a gentrification risk score (0–1).

    Args:
        input_size   : number of features per timestep
        hidden_size  : LSTM hidden units
        num_layers   : stacked LSTM layers
        dropout      : dropout between LSTM layers
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        returns: (batch, 1) risk score in [0, 1]
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention pooling over timesteps
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (attn_weights * lstm_out).sum(dim=1)                  # (batch, hidden)

        return self.classifier(context)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
