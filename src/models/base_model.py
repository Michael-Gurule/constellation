"""
Base model class with common functionality.
"""

import pickle
from pathlib import Path
from typing import Dict, Any
import json

from src.utils.logging_config import setup_logger
from config.settings import MODELS_DIR

logger = setup_logger(__name__)


class BaseModel:
    """
    Base class for all ML models.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.trained = False
        self.metrics = {}
        
        logger.info(f"Initialized {model_name}")
    
    def save(self, filepath: Path = None) -> Path:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model. If None, uses default.
        
        Returns:
            Path where model was saved
        """
        if filepath is None:
            filepath = MODELS_DIR / f"{self.model_name}.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
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
    
    def load(self, filepath: Path) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
        
        Returns:
            Self
        """
        # Load model
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.trained = metadata.get('trained', False)
                self.metrics = metadata.get('metrics', {})
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        return self.metrics