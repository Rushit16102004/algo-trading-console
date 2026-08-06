# PASSKEY: rushit2712
import os
import pandas as pd
from backend_engine.config import GBM_THRESHOLD, TCN_THRESHOLD
import importlib
live_prediction_engine_module = importlib.import_module("243A.live_prediction_engine")
LivePredictionEngine = live_prediction_engine_module.LivePredictionEngine

class Strategy243A:
    def __init__(self):
        self.name = "243A Strategy"
        self.description = "Consensus of LightGBM + TCN predictions filtered by HMM Regime"
        self._engine = None

    def get_engine(self):
        if self._engine is None:
            self._engine = LivePredictionEngine()
        return self._engine

    def predict(self, df: pd.DataFrame, *args, **kwargs) -> dict:
        """
        Takes the historical candles dataframe (5-minute interval).
        Returns a dict: {
            'signal': 1 (BUY), -1 (SELL), 0 (HOLD),
            'metrics': dict of calculated feature values / predictions
        }
        """
        engine = self.get_engine()
        try:
            predictions = engine.predict_latest(df)
            hmm_regime = predictions.get('hmm_regime_name', 'Unknown')
        except Exception as e:
            return {
                'signal': 0,
                'metrics': {'error': str(e), 'hmm_regime': 'Unknown'}
            }
            
        gbm_buy_latest = (predictions.get('gbm_predicted') == 1) and (predictions.get('gbm_prob_buy', 0) >= GBM_THRESHOLD)
        gbm_sell_latest = (predictions.get('gbm_predicted') == 0) and (predictions.get('gbm_prob_sell', 0) >= GBM_THRESHOLD)
        tcn_buy_latest = (predictions.get('tcn_predicted') == 'BUY') and (predictions.get('tcn_prob_buy', 0) >= TCN_THRESHOLD)
        tcn_sell_latest = (predictions.get('tcn_predicted') == 'SELL') and (predictions.get('tcn_prob_sell', 0) >= TCN_THRESHOLD)

        buy_consensus = gbm_buy_latest and tcn_buy_latest
        sell_consensus = gbm_sell_latest and tcn_sell_latest
        
        if buy_consensus and sell_consensus:
            consensus = 'HOLD'
        elif buy_consensus:
            consensus = 'BUY'
        elif sell_consensus:
            consensus = 'SELL'
        else:
            consensus = 'HOLD'

        blocked_regimes_buy = {'compression', 'expansiondown', 'distributiondown', 'markdown'}
        blocked_regimes_sell = {'compression', 'expansionup', 'distributionup', 'markup'}
        
        signal = 0
        if consensus == 'BUY' and hmm_regime not in blocked_regimes_buy:
            signal = 1
        elif consensus == 'SELL' and hmm_regime not in blocked_regimes_sell:
            signal = -1

        metrics = {
            'hmm_regime': hmm_regime,
            'gbm_predicted': predictions.get('gbm_predicted'),
            'gbm_prob_buy': round(predictions.get('gbm_prob_buy', 0), 4),
            'gbm_prob_sell': round(predictions.get('gbm_prob_sell', 0), 4),
            'tcn_predicted': predictions.get('tcn_predicted'),
            'tcn_prob_buy': round(predictions.get('tcn_prob_buy', 0), 4),
            'tcn_prob_sell': round(predictions.get('tcn_prob_sell', 0), 4),
            'consensus': consensus
        }
        
        return {
            'signal': signal,
            'metrics': metrics
        }
