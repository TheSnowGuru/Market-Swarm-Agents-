from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name, risk_tolerance=0.5, trade_frequency='medium'):
        self.name = name
        self.risk_tolerance = risk_tolerance
        self.trade_frequency = trade_frequency
        self.positions = []
        self.performance = 0

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def execute_trade(self):
        pass

    def process_event(self, event):
        raise NotImplementedError("This method should be overridden in subclasses")

    def take_position(self, signal):
        self.positions.append(signal)

    def evaluate_risk(self, market_data):
        # Implement basic risk evaluation
        return True

    def get_performance(self):
        return self.performance

    @abstractmethod
    def train(self, data):
        pass
