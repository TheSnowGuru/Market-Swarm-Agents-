import csv
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any

class EventType(Enum):
    """
    Comprehensive event types for trading strategies
    """
    # Base events
    TRADE_ENTRY = auto()
    TRADE_EXIT = auto()
    SIGNAL_GENERATED = auto()
    
    # Optimal Trade specific events
    OPTIMAL_TRADE_SIGNAL = auto()
    STOP_LOSS_TRIGGERED = auto()
    TAKE_PROFIT_TRIGGERED = auto()
    
    # Performance events
    PERFORMANCE_UPDATE = auto()
    RISK_THRESHOLD_BREACH = auto()
    
    # System events
    AGENT_INITIALIZATION = auto()
    AGENT_SHUTDOWN = auto()
    ERROR_EVENT = auto()

class EventStream:
    def __init__(self, file_path: str):
        """
        Initialize event logging stream
        
        Args:
            file_path (str): Path to event log file
        """
        self.file_path = file_path

    def log_event(self, 
                  agent: str, 
                  event_type: EventType, 
                  details: Dict[str, Any] = None):
        """
        Log a trading event
        
        Args:
            agent (str): Name of the agent generating the event
            event_type (EventType): Type of event
            details (dict, optional): Additional event details
        """
        if details is None:
            details = {}
        
        event_details = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent,
            'event_type': event_type.name,
            'details': str(details)
        }
        
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=event_details.keys())
            
            # Write headers if file is empty
            if file.tell() == 0:
                writer.writeheader()
            
            writer.writerow(event_details)

    def get_events(self, 
                   agent: str = None, 
                   event_type: EventType = None) -> list:
        """
        Retrieve logged events with optional filtering
        
        Args:
            agent (str, optional): Filter by agent name
            event_type (EventType, optional): Filter by event type
        
        Returns:
            list: Filtered events
        """
        events = []
        with open(self.file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if (not agent or row['agent'] == agent) and \
                   (not event_type or row['event_type'] == event_type.name):
                    events.append(row)
        return events
