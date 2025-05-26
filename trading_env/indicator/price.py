"""
Price Indicator

The Price indicator represents the current price of an asset.
Values are normalized to a range of 0-100 using Min-Max normalization.

Parameters:
    period (int): The period for price calculation (default: 20)
    min_price (float): Minimum price for normalization (default: None, will use historical min)
    max_price (float): Maximum price for normalization (default: None, will use historical max)

Returns:
    float: Normalized price value between 0 and 100
    * Higher values indicate higher price levels
    * Lower values indicate lower price levels
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

class Price:
    def __init__(self, period: int = 20, min_price: Optional[float] = None, max_price: Optional[float] = None):
        """
        Initialize Price indicator.

        Args:
            period (int): The period for price calculation (default: 20)
            min_price (float): Minimum price for normalization (default: None)
            max_price (float): Maximum price for normalization (default: None)
        """
        self.period = period
        self.min_price = min_price
        self.max_price = max_price
        self.price_history = []

    def calculate(self, data: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate normalized price value.

        Args:
            data (Union[pd.Series, np.ndarray]): Price data

        Returns:
            float: Normalized price value between 0 and 100
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        # Calculate average price over the period
        avg_price = data.rolling(window=self.period).mean().iloc[-1]
        
        # Update price history
        self.price_history.append(avg_price)
        
        # Determine min and max prices for normalization
        min_price = self.min_price if self.min_price is not None else min(self.price_history)
        max_price = self.max_price if self.max_price is not None else max(self.price_history)
        
        # Normalize price to 0-100 range
        normalized_price = ((avg_price - min_price) / (max_price - min_price)) * 100
        
        return normalized_price 