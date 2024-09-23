import pandas as pd

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.events = []

    def log_event(self, event):
        self.events.append(event)

    def save_logs(self):
        df = pd.DataFrame(self.events)
        df.to_csv(self.log_file, index=False)
