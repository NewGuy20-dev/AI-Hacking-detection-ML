"""Time-series anomaly detection with seasonal pattern recognition."""
import numpy as np
from collections import deque
from datetime import datetime


class TimeSeriesDetector:
    """Detect anomalies in time-series traffic patterns."""
    
    def __init__(self, window_size: int = 168):  # 168 hours = 1 week
        self.window_size = window_size
        self.hourly_baseline = {}  # hour -> [values]
        self.daily_baseline = {}   # day_of_week -> [values]
        self.history = deque(maxlen=window_size)
    
    def update(self, value: float, timestamp: datetime = None):
        """Add observation and update baselines."""
        ts = timestamp or datetime.now()
        hour = ts.hour
        day = ts.weekday()
        
        self.history.append({'value': value, 'hour': hour, 'day': day, 'ts': ts})
        
        # Update hourly baseline
        if hour not in self.hourly_baseline:
            self.hourly_baseline[hour] = []
        self.hourly_baseline[hour].append(value)
        if len(self.hourly_baseline[hour]) > 100:
            self.hourly_baseline[hour] = self.hourly_baseline[hour][-100:]
        
        # Update daily baseline
        if day not in self.daily_baseline:
            self.daily_baseline[day] = []
        self.daily_baseline[day].append(value)
        if len(self.daily_baseline[day]) > 100:
            self.daily_baseline[day] = self.daily_baseline[day][-100:]
    
    def detect(self, value: float, timestamp: datetime = None) -> dict:
        """Check if value is anomalous given time context."""
        ts = timestamp or datetime.now()
        hour, day = ts.hour, ts.weekday()
        
        result = {'value': value, 'hour': hour, 'day': day, 'anomalies': []}
        
        # Check against hourly baseline
        if hour in self.hourly_baseline and len(self.hourly_baseline[hour]) >= 5:
            hourly_mean = np.mean(self.hourly_baseline[hour])
            hourly_std = np.std(self.hourly_baseline[hour]) + 1e-8
            z_score = abs(value - hourly_mean) / hourly_std
            if z_score > 3:
                result['anomalies'].append({'type': 'hourly', 'z_score': round(z_score, 2)})
        
        # Check against daily baseline
        if day in self.daily_baseline and len(self.daily_baseline[day]) >= 5:
            daily_mean = np.mean(self.daily_baseline[day])
            daily_std = np.std(self.daily_baseline[day]) + 1e-8
            z_score = abs(value - daily_mean) / daily_std
            if z_score > 3:
                result['anomalies'].append({'type': 'daily', 'z_score': round(z_score, 2)})
        
        # Check for sudden spike (vs recent history)
        if len(self.history) >= 10:
            recent = [h['value'] for h in list(self.history)[-10:]]
            recent_mean = np.mean(recent)
            if value > recent_mean * 3:
                result['anomalies'].append({'type': 'spike', 'ratio': round(value/recent_mean, 2)})
        
        result['is_anomaly'] = len(result['anomalies']) > 0
        return result


if __name__ == '__main__':
    detector = TimeSeriesDetector()
    
    # Build baseline (normal traffic ~100 req/min)
    from datetime import timedelta
    base_time = datetime.now()
    for i in range(50):
        detector.update(100 + np.random.randn()*10, base_time + timedelta(hours=i))
    
    # Test normal
    result = detector.detect(105)
    print(f"Normal (105): anomaly={result['is_anomaly']}")
    
    # Test anomaly
    result = detector.detect(500)
    print(f"Spike (500): anomaly={result['is_anomaly']}, reasons={result['anomalies']}")
