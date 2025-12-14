"""Predictive threat modeling and risk forecasting."""
import numpy as np
from collections import deque
from datetime import datetime, timedelta


class PredictiveModel:
    """Forecast attack likelihood based on historical patterns."""
    
    def __init__(self, history_days: int = 30):
        self.history = deque(maxlen=history_days * 24)  # Hourly data
        self.daily_patterns = {}  # day_of_week -> attack_rate
        self.hourly_patterns = {}  # hour -> attack_rate
        self.trend = 0  # Overall trend direction
    
    def record(self, timestamp: datetime, attack_count: int, total_traffic: int):
        """Record hourly attack statistics."""
        hour = timestamp.hour
        day = timestamp.weekday()
        rate = attack_count / max(total_traffic, 1)
        
        self.history.append({
            'ts': timestamp, 'attacks': attack_count, 
            'traffic': total_traffic, 'rate': rate,
            'hour': hour, 'day': day
        })
        
        # Update patterns
        if hour not in self.hourly_patterns:
            self.hourly_patterns[hour] = []
        self.hourly_patterns[hour].append(rate)
        
        if day not in self.daily_patterns:
            self.daily_patterns[day] = []
        self.daily_patterns[day].append(rate)
        
        # Calculate trend
        if len(self.history) >= 48:
            recent = [h['rate'] for h in list(self.history)[-24:]]
            older = [h['rate'] for h in list(self.history)[-48:-24]]
            self.trend = np.mean(recent) - np.mean(older)
    
    def forecast(self, target_time: datetime = None, hours_ahead: int = 24) -> list:
        """Forecast attack probability for upcoming hours."""
        if len(self.history) < 24:
            return [{'error': 'insufficient_data'}]
        
        start = target_time or datetime.now()
        forecasts = []
        
        for h in range(hours_ahead):
            future_time = start + timedelta(hours=h)
            hour = future_time.hour
            day = future_time.weekday()
            
            # Base rate from patterns
            hourly_rate = np.mean(self.hourly_patterns.get(hour, [0.1]))
            daily_rate = np.mean(self.daily_patterns.get(day, [0.1]))
            
            # Combined forecast with trend
            base_forecast = 0.6 * hourly_rate + 0.4 * daily_rate
            adjusted = base_forecast + self.trend * (h / 24)  # Trend adjustment
            
            risk_level = 'LOW' if adjusted < 0.1 else 'MEDIUM' if adjusted < 0.3 else 'HIGH'
            
            forecasts.append({
                'time': future_time.isoformat(),
                'hour': hour,
                'attack_probability': round(min(max(adjusted, 0), 1), 3),
                'risk_level': risk_level
            })
        
        return forecasts
    
    def get_risk_windows(self, threshold: float = 0.3) -> list:
        """Identify high-risk time windows."""
        forecast = self.forecast(hours_ahead=168)  # 1 week
        high_risk = [f for f in forecast if f.get('attack_probability', 0) >= threshold]
        return high_risk


if __name__ == '__main__':
    model = PredictiveModel()
    
    # Simulate historical data (higher attacks at night)
    base = datetime.now() - timedelta(days=7)
    for d in range(7):
        for h in range(24):
            ts = base + timedelta(days=d, hours=h)
            # More attacks at night (0-6) and weekends
            attack_mult = 3 if h < 6 else 1
            if ts.weekday() >= 5:
                attack_mult *= 2
            attacks = int(10 * attack_mult + np.random.randint(0, 5))
            model.record(ts, attacks, 1000)
    
    print("24-Hour Forecast:")
    forecast = model.forecast(hours_ahead=6)
    for f in forecast:
        print(f"  {f['time'][-8:-3]}: prob={f['attack_probability']:.3f} [{f['risk_level']}]")
    
    print(f"\nHigh-risk windows (next week): {len(model.get_risk_windows())} periods")
