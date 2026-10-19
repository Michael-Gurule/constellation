"""
Load and test LSTM model trained in Colab.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from constellation.legacy.settings import MODELS_DIR


class LSTMForecaster(nn.Module):
    """LSTM model architecture (must match Colab definition)."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


def load_colab_model(model_path: Path):
    """
    Load model

    Args:
        model_path: Path to saved model

    Returns:
        Tuple of (model, scaler, metadata)
    """
    print(f"Loading model from {model_path}...")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    # Extract components
    input_dim = model_data["input_dim"]
    hidden_dim = model_data["hidden_dim"]
    num_layers = model_data["num_layers"]
    scaler = model_data["scaler"]

    # Reconstruct model
    model = LSTMForecaster(input_dim, hidden_dim, num_layers)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    print("✓ Model loaded successfully")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Sequence length: {model_data['sequence_length']}")
    print(f"  Forecast horizon: {model_data['forecast_horizon']}")

    return model, scaler, model_data


def main():
    # Load model
    colab_model_path = MODELS_DIR / "degradation_forecaster_colab.pkl"

    if not colab_model_path.exists():
        print(f"ERROR: Model not found at {colab_model_path}")
        print("Please download it from Colab and place it in data/models/")
        return

    model, _scaler, metadata = load_colab_model(colab_model_path)

    # Test with synthetic data
    print("\nTesting model with dummy data...")

    input_dim = metadata["input_dim"]
    seq_len = metadata["sequence_length"]

    # Create dummy sequence
    dummy_sequence = np.random.randn(1, seq_len, input_dim).astype(np.float32)
    dummy_tensor = torch.FloatTensor(dummy_sequence)

    # Make prediction
    with torch.no_grad():
        prediction = model(dummy_tensor)

    print(f"✓ Test prediction: {prediction.item():.4f}")
    print("\n✓ Model is ready to use!")

    # Print feature columns
    print(f"\nRequired features ({len(metadata['feature_cols'])}):")
    for i, col in enumerate(metadata["feature_cols"][:10], 1):
        print(f"  {i}. {col}")
    if len(metadata["feature_cols"]) > 10:
        print(f"  ... and {len(metadata['feature_cols']) - 10} more")


if __name__ == "__main__":
    main()
