import csv
from datetime import datetime

class EventStream:
    def __init__(self, file_path):
        self.file_path = file_path

    def log_event(self, agent, event_type, details):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), agent, event_type, details])

    def get_events(self):
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            return list(reader)
