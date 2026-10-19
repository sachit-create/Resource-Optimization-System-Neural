from collections import deque
class SlidingWindowBuffer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    def add(self, data_point):
        """
        data_point: list or numpy array of features
        """
        self.buffer.append(data_point)
    def is_ready(self):
        return len(self.buffer) == self.window_size
    def get_sequence(self):
        """
        Returns data in correct time order for LSTM
        Shape: (1, window_size, num_features)
        """
        if not self.is_ready():
            return None
        return list(self.buffer)
