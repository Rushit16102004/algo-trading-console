# PASSKEY: rushit2712
import numpy as np
import pandas as pd

class StrategyLongpineZFTF:
    def __init__(self):
        self.name = "Longpine ZFTF Strategy"
        self.description = "Z-Score Trend Following Strategy (20-period). Enters LONG on zscore > 2.0 & slope > 0.001."
        self.window = 20
        self.z_threshold = 2.0
        self.slope_threshold = 0.001

    def predict(self, df: pd.DataFrame, in_position: bool = False, *args, **kwargs) -> dict:
        """
        Z Score Trend Following strategy using PineScript version:
        - window = 20
        - zscore = (close - mean_20) / std_20
        - slope = linreg slope (window 20)
        - Enters LONG if zscore > 2.0 and slope > 0.001
        """
        if len(df) < self.window:
            return {'signal': 0, 'metrics': {'z_score': 0.0, 'slope': 0.0}}
            
        close_slice = df['close'].tail(self.window)
        mean_val = float(close_slice.mean())
        std_val = float(close_slice.std())
        
        z_val = ((df['close'].iloc[-1] - mean_val) / std_val) if std_val > 0 else 0.0
        
        # Calculate linear regression slope
        y = close_slice.values
        x = np.arange(self.window)
        x_mean = x.mean()
        x_dev = x - x_mean
        x_var = (x_dev**2).sum()
        slope_val = np.dot(x_dev, y) / x_var if x_var > 0 else 0.0
            
        signal = 0
        if not in_position:
            if std_val > 0:
                if z_val > self.z_threshold and slope_val > self.slope_threshold:
                    signal = 1
                    
        metrics = {
            'z_score': round(z_val, 4),
            'slope': round(slope_val, 6)
        }
        
        return {
            'signal': signal,
            'metrics': metrics
        }
