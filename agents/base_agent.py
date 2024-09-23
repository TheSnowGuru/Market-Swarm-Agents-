from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name):
        self.name = name
        self.performance = 0

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def execute_trade(self):
        pass

    def get_performance(self):
        return self.performance

    @abstractmethod
    def train(self, data):
        pass
