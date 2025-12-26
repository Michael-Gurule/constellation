"""
Lightstreamer client for connecting to NASA's ISS telemetry feed.
Handles real-time streaming data from the International Space Station.
"""

import time
import json
import requests
from typing import Dict, List, Callable, Optional
from datetime import datetime
import threading
from queue import Queue

from config.settings import (
    LIGHTSTREAMER_URL, 
    LIGHTSTREAMER_ADAPTER_SET,
    MONITORED_PARAMS
)
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class LightstreamerClient:
    """
    Client for NASA's Lightstreamer telemetry feed.
    Implements a simplified Lightstreamer protocol for ISS data.
    """
    
    def __init__(self, callback: Optional[Callable] = None):
        """
        Initialize Lightstreamer client.
        
        Args:
            callback: Function to call when new data arrives.
                     Signature: callback(data: Dict[str, any])
        """
        self.base_url = LIGHTSTREAMER_URL
        self.adapter_set = LIGHTSTREAMER_ADAPTER_SET
        self.session_id = None
        self.callback = callback
        self.running = False
        self.data_queue = Queue()
        self._thread = None
        
        logger.info("Lightstreamer client initialized")
    
    def connect(self) -> bool:
        """
        Establish connection to Lightstreamer server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create session
            session_url = f"{self.base_url}/lightstreamer/create_session.txt"
            
            params = {
                "LS_adapter_set": self.adapter_set,
                "LS_polling": "true",
                "LS_polling_millis": 0,
                "LS_idle_millis": 0,
            }
            
            response = requests.post(session_url, data=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Failed to create session: {response.status_code}")
                return False
            
            # Parse session ID from response
            for line in response.text.split('\n'):
                if line.startswith('SessionId:'):
                    self.session_id = line.split(':')[1].strip()
                    logger.info(f"Session created: {self.session_id}")
                    return True
            
            logger.error("No session ID in response")
            return False
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def subscribe(self, parameters: List[str] = None) -> bool:
        """
        Subscribe to telemetry parameters.
        
        Args:
            parameters: List of parameter IDs to subscribe to.
                       If None, subscribes to all MONITORED_PARAMS.
        
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.session_id:
            logger.error("Not connected. Call connect() first.")
            return False
        
        if parameters is None:
            parameters = MONITORED_PARAMS
        
        try:
            subscribe_url = f"{self.base_url}/lightstreamer/control.txt"
            
            # Build subscription request
            items = " ".join(parameters)
            fields = "TimeStamp Value Status"
            
            params = {
                "LS_session": self.session_id,
                "LS_op": "add",
                "LS_table": "1",
                "LS_id": items,
                "LS_schema": fields,
                "LS_mode": "MERGE",
            }
            
            response = requests.post(subscribe_url, data=params, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Subscribed to {len(parameters)} parameters")
                return True
            else:
                logger.error(f"Subscription failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            return False
    
    def _poll_data(self):
        """
        Poll for new data (runs in separate thread).
        """
        poll_url = f"{self.base_url}/lightstreamer/bind_session.txt"
        
        params = {
            "LS_session": self.session_id,
        }
        
        while self.running:
            try:
                response = requests.post(
                    poll_url, 
                    data=params, 
                    timeout=30,
                    stream=True
                )
                
                for line in response.iter_lines():
                    if not self.running:
                        break
                    
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        # Parse update messages
                        if decoded_line.startswith('U,'):
                            self._process_update(decoded_line)
                
            except requests.exceptions.Timeout:
                logger.debug("Poll timeout, reconnecting...")
                continue
            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(5)  # Wait before retry
    
    def _process_update(self, message: str):
        """
        Process an update message from the stream.
        
        Args:
            message: Raw update message from Lightstreamer
        """
        try:
            # Parse message format: U,<table>,<item>,<field1>|<field2>|...
            parts = message.split(',', 3)
            
            if len(parts) < 4:
                return
            
            item_index = parts[2]
            fields = parts[3].split('|')
            
            # Extract timestamp, value, status
            if len(fields) >= 2:
                timestamp_str = fields[0]
                value_str = fields[1]
                
                # Build data record
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'parameter_id': item_index,
                    'value': value_str,
                    'raw_message': message
                }
                
                # Add to queue
                self.data_queue.put(data)
                
                # Call callback if provided
                if self.callback:
                    self.callback(data)
                    
        except Exception as e:
            logger.error(f"Error processing update: {e}")
    
    def start_streaming(self):
        """
        Start streaming data in background thread.
        """
        if self.running:
            logger.warning("Already streaming")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._poll_data, daemon=True)
        self._thread.start()
        logger.info("Started streaming data")
    
    def stop_streaming(self):
        """
        Stop streaming data.
        """
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stopped streaming data")
    
    def get_data(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get next data item from queue (blocking).
        
        Args:
            timeout: Maximum time to wait for data
        
        Returns:
            Data dictionary or None if timeout
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except:
            return None
    
    def disconnect(self):
        """
        Disconnect from Lightstreamer server.
        """
        self.stop_streaming()
        
        if self.session_id:
            try:
                destroy_url = f"{self.base_url}/lightstreamer/control.txt"
                params = {
                    "LS_session": self.session_id,
                    "LS_op": "destroy"
                }
                requests.post(destroy_url, data=params, timeout=5)
                logger.info("Session destroyed")
            except Exception as e:
                logger.error(f"Error destroying session: {e}")
            
            self.session_id = None