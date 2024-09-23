from agents.agent_base import BaseAgent

class TrendFollowerAgent(BaseAgent):
    def process_event(self, event):
        # Implement trend-following strategy
        if event['trend'] == "up":
            signal = {"type": "buy", "amount": 10}
            self.take_position(signal)
