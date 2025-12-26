"""
Main script for collecting ISS telemetry data.
Run this to start streaming and storing telemetry.
"""

import time
import signal
import sys
from datetime import datetime
from typing import List, Dict

from src.ingestion.lightstreamer_client import LightstreamerClient
from src.ingestion.data_validator import TelemetryValidator
from src.ingestion.storage_handler import LocalStorageHandler
from config.settings import MONITORED_PARAMS
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class TelemetryCollector:
    """
    Orchestrates telemetry collection, validation, and storage.
    """
    
    def __init__(self, batch_size: int = 100, save_interval: int = 300):
        """
        Initialize collector.
        
        Args:
            batch_size: Number of records to accumulate before validation
            save_interval: Seconds between saves to disk
        """
        self.batch_size = batch_size
        self.save_interval = save_interval
        
        self.client = LightstreamerClient(callback=self._on_data_received)
        self.validator = TelemetryValidator()
        self.storage = LocalStorageHandler()
        
        self.buffer: List[Dict] = []
        self.last_save_time = time.time()
        self.running = False
        self.total_received = 0
        self.total_saved = 0
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Telemetry collector initialized")
    
    def _on_data_received(self, data: Dict):
        """
        Callback when new data arrives from Lightstreamer.
        
        Args:
            data: Telemetry data dictionary
        """
        self.total_received += 1
        
        # Validate data
        is_valid, error_msg = self.validator.validate_record(data)
        
        if is_valid:
            self.buffer.append(data)
            
            # Check if we should save
            if len(self.buffer) >= self.batch_size or \
               (time.time() - self.last_save_time) >= self.save_interval:
                self._save_buffer()
        else:
            logger.warning(f"Invalid data: {error_msg}")
    
    def _save_buffer(self):
        """
        Save buffered records to storage.
        """
        if not self.buffer:
            return
        
        # Group by subsystem for organized storage
        attitude_records = []
        comm_records = []
        
        for record in self.buffer:
            param_info = self.validator.get_parameter_info(record['parameter_id'])
            if param_info:
                if param_info['subsystem'] == 'attitude_control':
                    attitude_records.append(record)
                elif param_info['subsystem'] == 'communications':
                    comm_records.append(record)
        
        # Save by subsystem
        if attitude_records:
            self.storage.save_batch(attitude_records, subsystem='attitude_control')
            self.total_saved += len(attitude_records)
        
        if comm_records:
            self.storage.save_batch(comm_records, subsystem='communications')
            self.total_saved += len(comm_records)
        
        logger.info(
            f"Saved batch: {len(self.buffer)} records "
            f"(Total received: {self.total_received}, saved: {self.total_saved})"
        )
        
        # Clear buffer
        self.buffer.clear()
        self.last_save_time = time.time()
    
    def start(self):
        """
        Start collecting telemetry.
        """
        logger.info("Starting telemetry collection...")
        
        # Connect to Lightstreamer
        if not self.client.connect():
            logger.error("Failed to connect to Lightstreamer")
            return False
        
        # Subscribe to parameters
        if not self.client.subscribe(MONITORED_PARAMS):
            logger.error("Failed to subscribe to parameters")
            return False
        
        # Start streaming
        self.client.start_streaming()
        self.running = True
        
        logger.info("Telemetry collection started. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1)
                
                # Periodic status update
                if self.total_received % 1000 == 0 and self.total_received > 0:
                    logger.info(f"Status: {self.total_received} received, {self.total_saved} saved")
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """
        Stop collecting telemetry and cleanup.
        """
        logger.info("Stopping telemetry collection...")
        
        self.running = False
        
        # Save any remaining buffered data
        self._save_buffer()
        
        # Disconnect from Lightstreamer
        self.client.disconnect()
        
        logger.info(
            f"Collection stopped. Final stats: "
            f"{self.total_received} received, {self.total_saved} saved"
        )
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully.
        """
        logger.info(f"Received signal {signum}")
        self.running = False


def main():
    """
    Main entry point for telemetry collection.
    """
    print("=" * 60)
    print("CONSTELLATION - ISS Telemetry Collection")
    print("=" * 60)
    print()
    
    collector = TelemetryCollector(
        batch_size=100,
        save_interval=300  # Save every 5 minutes
    )
    
    collector.start()


if __name__ == "__main__":
    main()