"""
Data validation utilities for ISS telemetry.
Ensures data quality and identifies anomalies in incoming telemetry.
"""

import json
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime

from config.settings import PROJECT_ROOT
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class TelemetryValidator:
    """
    Validates incoming telemetry data against expected ranges and types.
    """
    
    def __init__(self, params_file: Path = None):
        """
        Initialize validator with parameter definitions.
        
        Args:
            params_file: Path to telemetry_params.json. 
                        If None, uses default from config.
        """
        if params_file is None:
            params_file = PROJECT_ROOT / "config" / "telemetry_params.json"
        
        with open(params_file, 'r') as f:
            self.params_config = json.load(f)
        
        logger.info(f"Loaded {len(self.params_config)} parameter definitions")
    
    def validate_record(self, record: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate a single telemetry record.
        
        Args:
            record: Dictionary with 'parameter_id', 'value', 'timestamp'
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        required_fields = ['parameter_id', 'value', 'timestamp']
        for field in required_fields:
            if field not in record:
                return False, f"Missing required field: {field}"
        
        param_id = record['parameter_id']
        value_str = record['value']
        
        # Check if parameter is known
        if param_id not in self.params_config:
            return False, f"Unknown parameter ID: {param_id}"
        
        param_config = self.params_config[param_id]
        
        # Validate data type
        try:
            if param_config['type'] == 'float':
                value = float(value_str)
            elif param_config['type'] == 'int':
                value = int(value_str)
            else:
                value = value_str
        except ValueError:
            return False, f"Invalid type for {param_id}: expected {param_config['type']}"
        
        # Validate range (for numeric types)
        if param_config['type'] in ['float', 'int'] and 'normal_range' in param_config:
            min_val, max_val = param_config['normal_range']
            if not (min_val <= value <= max_val):
                # This is a warning, not an error - could be legitimate anomaly
                logger.warning(
                    f"Value {value} for {param_id} outside normal range "
                    f"[{min_val}, {max_val}]"
                )
        
        return True, None
    
    def validate_batch(self, records: list) -> Dict[str, any]:
        """
        Validate a batch of telemetry records.
        
        Args:
            records: List of telemetry record dictionaries
        
        Returns:
            Dictionary with validation statistics
        """
        total = len(records)
        valid = 0
        invalid = 0
        errors = []
        
        for record in records:
            is_valid, error_msg = self.validate_record(record)
            if is_valid:
                valid += 1
            else:
                invalid += 1
                errors.append({
                    'record': record,
                    'error': error_msg
                })
        
        stats = {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'validation_rate': valid / total if total > 0 else 0,
            'errors': errors
        }
        
        logger.info(
            f"Validated {total} records: {valid} valid, {invalid} invalid "
            f"({stats['validation_rate']:.1%} pass rate)"
        )
        
        return stats
    
    def get_parameter_info(self, param_id: str) -> Optional[Dict]:
        """
        Get configuration for a specific parameter.
        
        Args:
            param_id: Parameter identifier
        
        Returns:
            Parameter configuration dictionary or None
        """
        return self.params_config.get(param_id)
    
    def is_out_of_range(self, param_id: str, value: float) -> bool:
        """
        Check if a value is outside normal operating range.
        
        Args:
            param_id: Parameter identifier
            value: Numeric value to check
        
        Returns:
            True if out of range, False otherwise
        """
        if param_id not in self.params_config:
            return False
        
        param_config = self.params_config[param_id]
        
        if 'normal_range' not in param_config:
            return False
        
        min_val, max_val = param_config['normal_range']
        return not (min_val <= value <= max_val)