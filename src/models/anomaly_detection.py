"""
Anomaly detection models for telemetry data.
Implements Isolation Forest and LSTM Autoencoder.
"""

import numpy as np
import pandas as pd
from pathlib import Path 
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional

from src.models.base_model import BaseModel
from src.utils.logging_config import setup_logger
from src.utils.metrics import calculate_anomaly_metrics
from config.settings import MODELS_DIR

logger = setup_logger(__name__)


class IsolationForestDetector(BaseModel):
    """
    Isolation Forest for anomaly detection.
    Fast, scalable anomaly detection for high-dimensional data.
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
        """
        super().__init__("isolation_forest")
        
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X: pd.DataFrame) -> 'IsolationForestDetector':
        """
        Train Isolation Forest on normal data.
        
        Args:
            X: Feature matrix (assumes mostly normal data)
        
        Returns:
            Self
        """
        logger.info(f"Training Isolation Forest on {len(X)} samples")
        
        # Handle NaN values
        X_clean = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train model
        self.model.fit(X_scaled)
        
        self.trained = True
        logger.info("Training complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predictions (1 = normal, -1 = anomaly)
        """
        X_clean = X.fillna(X.median())
        X_scaled = self.scaler.transform(X_clean)
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous).
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of anomaly scores
        """
        X_clean = X.fillna(X.median())
        X_scaled = self.scaler.transform(X_clean)
        
        scores = self.model.score_samples(X_scaled)
        return scores
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y_true: True labels (0 = normal, 1 = anomaly)
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
        y_pred = (predictions == -1).astype(int)
        
        self.metrics = calculate_anomaly_metrics(y_true, y_pred)
        
        logger.info(f"Evaluation metrics: {self.metrics}")
        return self.metrics
    
    def save(self, filepath: Path = None) -> Path:
        """
        Save model and scaler to disk.
        
        Args:
            filepath: Path to save model
        
        Returns:
            Path where model was saved
        """
        if filepath is None:
            filepath = MODELS_DIR / f"{self.model_name}.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both model and scaler
        import pickle
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save metadata
        import json
        metadata = {
            'model_name': self.model_name,
            'trained': self.trained,
            'metrics': self.metrics
        }
        
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load(self, filepath: Path) -> 'IsolationForestDetector':
        """
        Load model and scaler from disk.
        
        Args:
            filepath: Path to saved model
        
        Returns:
            Self
        """
        import pickle
        import json
        
        # Load model and scaler
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.contamination = save_data.get('contamination', 0.1)
        self.n_estimators = save_data.get('n_estimators', 100)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.trained = metadata.get('trained', False)
                self.metrics = metadata.get('metrics', {})
        
        logger.info(f"Model loaded from {filepath}")
        return self

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder neural network for anomaly detection.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        """
        Initialize LSTM Autoencoder.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        batch_size, seq_len, _ = x.shape
        
        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        encoded = self.encoder_fc(hidden.squeeze(0))
        
        # Decode
        decoded = self.decoder_fc(encoded)
        decoded = decoded.unsqueeze(1).repeat(1, seq_len, 1)
        reconstructed, _ = self.decoder_lstm(decoded)
        
        return reconstructed


class LSTMAutoencoderDetector(BaseModel):
    """
    LSTM Autoencoder for time series anomaly detection.
    """
    
    def __init__(
        self, 
        input_dim: int,
        sequence_length: int = 100,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM Autoencoder detector.
        
        Args:
            input_dim: Number of features
            sequence_length: Length of input sequences
            hidden_dim: Hidden layer size
            latent_dim: Latent space size
            learning_rate: Learning rate for training
        """
        super().__init__("lstm_autoencoder")
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # Initialize model
        self.model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # For anomaly detection
        self.threshold = None
    
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sliding window sequences.
        
        Args:
            data: Time series data (samples, features)
        
        Returns:
            Sequences (samples, sequence_length, features)
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def train(
        self, 
        X: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> 'LSTMAutoencoderDetector':
        """
        Train LSTM Autoencoder.
        
        Args:
            X: Feature matrix
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
        
        Returns:
            Self
        """
        logger.info(f"Training LSTM Autoencoder on {len(X)} samples")
        
        # Prepare data
        X_clean = X.fillna(X.median())
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Create sequences
        sequences = self.create_sequences(X_scaled)
        
        # Train/validation split
        split_idx = int(len(sequences) * (1 - validation_split))
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_sequences)
        val_tensor = torch.FloatTensor(val_sequences)
        
        # Create data loaders
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_reconstructed = self.model(val_tensor)
                    val_loss = self.criterion(val_reconstructed, val_tensor).item()
                
                logger.info(
                    f"Epoch {epoch}/{epochs}: "
                    f"Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}"
                )
                self.model.train()
        
        # Calculate anomaly threshold (95th percentile of reconstruction error)
        self.model.eval()
        with torch.no_grad():
            train_reconstructed = self.model(train_tensor)
            reconstruction_errors = torch.mean((train_tensor - train_reconstructed) ** 2, dim=(1, 2))
            self.threshold = np.percentile(reconstruction_errors.numpy(), 95)
        
        self.trained = True
        logger.info(f"Training complete. Anomaly threshold: {self.threshold:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predictions (0 = normal, 1 = anomaly)
        """
        scores = self.score_samples(X)
        predictions = (scores > self.threshold).astype(int)
        return predictions
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get reconstruction errors as anomaly scores.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of reconstruction errors
        """
        X_clean = X.fillna(X.median())
        X_scaled = self.scaler.transform(X_clean)
        
        # Create sequences
        sequences = self.create_sequences(X_scaled)
        sequences_tensor = torch.FloatTensor(sequences)
        
        # Get reconstruction errors
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(sequences_tensor)
            errors = torch.mean((sequences_tensor - reconstructed) ** 2, dim=(1, 2))
        
        return errors.numpy()
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y_true: True labels (0 = normal, 1 = anomaly)
        
        Returns:
            Dictionary of metrics
        """
        # Adjust y_true for sequence length
        y_true_sequences = y_true[self.sequence_length - 1:]
        
        predictions = self.predict(X)
        
        self.metrics = calculate_anomaly_metrics(y_true_sequences, predictions)
        
        logger.info(f"Evaluation metrics: {self.metrics}")
        return self.metrics