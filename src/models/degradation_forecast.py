"""
Degradation forecasting models.
LSTM and Temporal Fusion Transformer for predicting component degradation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

from src.models.base_model import BaseModel
from src.utils.logging_config import setup_logger
from src.utils.metrics import calculate_regression_metrics

logger = setup_logger(__name__)


class LSTMForecaster(nn.Module):
    """
    LSTM neural network for time series forecasting.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMForecaster, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


class DegradationForecaster(BaseModel):
    """
    LSTM-based degradation forecasting model.
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 100,
        forecast_horizon: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 2,
        learning_rate: float = 0.001
    ):
        """
        Initialize degradation forecaster.
        
        Args:
            input_dim: Number of features
            sequence_length: Length of input sequences
            forecast_horizon: Steps ahead to forecast
            hidden_dim: Hidden layer size
            num_layers: Number of LSTM layers
            learning_rate: Learning rate
        """
        super().__init__("degradation_forecaster")
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        
        # Initialize model
        self.model = LSTMForecaster(input_dim, hidden_dim, num_layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def create_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training.
        
        Args:
            data: Feature data
            target: Target variable
        
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        X_sequences = []
        y_targets = []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X_sequences.append(data[i:i + self.sequence_length])
            y_targets.append(target[i + self.sequence_length + self.forecast_horizon - 1])
        
        return np.array(X_sequences), np.array(y_targets)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> 'DegradationForecaster':
        """
        Train forecasting model.
        
        Args:
            X: Feature matrix
            y: Target variable (parameter to forecast)
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation fraction
        
        Returns:
            Self
        """
        logger.info(f"Training degradation forecaster on {len(X)} samples")
        
        # Prepare data
        X_clean = X.fillna(X.median())
        X_scaled = self.scaler.fit_transform(X_clean)
        y_values = y.values
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_values)
        
        # Train/validation split
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = self.criterion(val_predictions, y_val_tensor).item()
                
                logger.info(
                    f"Epoch {epoch}/{epochs}: "
                    f"Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}"
                )
                self.model.train()
        
        self.trained = True
        logger.info("Training complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predictions
        """
        X_clean = X.fillna(X.median())
        X_scaled = self.scaler.transform(X_clean)
        
        # Create sequences (without targets)
        sequences = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            sequences.append(X_scaled[i:i + self.sequence_length])
        
        sequences = np.array(sequences)
        sequences_tensor = torch.FloatTensor(sequences)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(sequences_tensor)
        
        return predictions.numpy().flatten()
    
    def forecast_future(
        self,
        X_recent: pd.DataFrame,
        steps: int = 30
    ) -> np.ndarray:
        """
        Forecast multiple steps into future.
        
        Args:
            X_recent: Recent feature data (at least sequence_length rows)
            steps: Number of steps to forecast
        
        Returns:
            Array of forecasted values
        """
        # Use last sequence_length rows
        X_clean = X_recent.iloc[-self.sequence_length:].fillna(X_recent.median())
        X_scaled = self.scaler.transform(X_clean)
        
        forecasts = []
        current_sequence = X_scaled.copy()
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(steps):
                # Predict next value
                sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
                prediction = self.model(sequence_tensor).item()
                forecasts.append(prediction)
                
                # Update sequence (slide window)
                # Note: This is simplified - in production, you'd need to update all features
                current_sequence = np.roll(current_sequence, -1, axis=0)
                # You would update the last row with new predicted values + other features
        
        return np.array(forecasts)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate forecaster performance.
        
        Args:
            X: Feature matrix
            y: True target values
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        # Align y_true with predictions (account for sequence length)
        y_true = y.values[self.sequence_length - 1:len(predictions) + self.sequence_length - 1]
        
        self.metrics = calculate_regression_metrics(y_true, predictions)
        
        logger.info(f"Evaluation metrics: {self.metrics}")
        return self.metrics