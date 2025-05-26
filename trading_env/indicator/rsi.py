"""
Relative Strength Index (RSI) Indicator

The RSI is a momentum oscillator that measures the speed and change of price movements.
It is already normalized to a range of 0-100.

Parameters:
    period (int): The period for RSI calculation (default: 14)
    overbought (float): The overbought threshold (default: 70)
    oversold (float): The oversold threshold (default: 30)

Returns:
    float: RSI value between 0 and 100
    * Values above 70 indicate overbought conditions
    * Values below 30 indicate oversold conditions
    * Values around 50 indicate neutral conditions
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

class RSI:
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Initialize RSI indicator.

        Args:
            period (int): The period for RSI calculation (default: 14)
            overbought (float): The overbought threshold (default: 70)
            oversold (float): The oversold threshold (default: 30)
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def calculate(self, data: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate RSI value.

        Args:
            data (Union[pd.Series, np.ndarray]): Price data

        Returns:
            float: RSI value between 0 and 100
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        # Calculate price changes
        delta = data.diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]  # Return the last RSI value 