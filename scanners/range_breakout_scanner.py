import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from utils.data_fetcher import DataFetcher
from utils.technical_indicators import TechnicalIndicators

class RangeBreakoutScanner:
    """Range Breakout Scanner using Pine Script logic with 4-hour intervals"""

    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.tech_indicators = TechnicalIndicators()
        self.ist = pytz.timezone('Asia/Kolkata')

    def scan(self, timeframe="1h", lookback_days=60):
        """
        Scan for range breakout signals using Pine Script logic
        
        Args:
            timeframe: Data timeframe (must be "1h" for proper 4h resampling)
            lookback_days: Number of days to look back

        Returns:
            DataFrame with range breakout signals
        """
        symbols = self.data_fetcher.get_nse_stock_list()
        results = []

        for symbol in symbols[:100]:  # Limit to 100 symbols for performance
            try:
                # Fetch 1h data
                data = self.data_fetcher.get_stock_data(
                    symbol,
                    period=f"{lookback_days}d",
                    interval=timeframe
                )

                if data is None or len(data) < 100:
                    continue

                # Ensure datetime index
                data.index = pd.to_datetime(data.index)

                # Convert to 4h timeframe
                data = data.resample("4h").agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

                if len(data) < 25:
                    continue

                # Calculate ATR
                atr = self.tech_indicators.calculate_atr(data, period=min(500, len(data)-1))
                current_atr = atr.iloc[-1]

                # Define recent range using last 20 bars
                recent = data.tail(21)
                range_high = recent['High'].iloc[:-1].max()
                range_low = recent['Low'].iloc[:-1].min()
                range_size = range_high - range_low
                current = recent.iloc[-1]
                prev = recent.iloc[-2]

                # Volume confirmation
                avg_vol = data['Volume'].tail(20).mean()
                vol_ratio = current['Volume'] / avg_vol

                # Breakout rules
                breakout_threshold = current_atr * 0.1
                min_range_size = current_atr * 2

                if range_size < min_range_size:
                    continue

                result = None

                # Bullish breakout
                if (current['Close'] > range_high + breakout_threshold and
                    current['High'] > range_high and
                    prev['Close'] <= range_high):

                    breakout_strength = ((current['Close'] - range_high) / current_atr) * 100
                    price_change = ((current['Close'] - prev['Close']) / prev['Close']) * 100
                    stop_loss = range_low
                    risk = current['Close'] - stop_loss
                    target = current['Close'] + (risk * 2)

                    result = {
                        'Symbol': symbol.replace('.NS', ''),
                        'Breakout_Type': 'Bullish Range Breakout',
                        'Breakout_Strength': round(breakout_strength, 2),
                        'Current_Price': round(current['Close'], 2),
                        'Range_High': round(range_high, 2),
                        'Range_Low': round(range_low, 2),
                        'Range_Size': round(range_size, 2),
                        'Price_Change_%': round(price_change, 2),
                        'Volume_Ratio': round(vol_ratio, 2),
                        'Stop_Loss': round(stop_loss, 2),
                        'Target': round(target, 2),
                        'Risk_Reward': '1:2',
                        'ATR': round(current_atr, 2),
                        'Scan_Time': datetime.now(self.ist).strftime('%H:%M:%S')
                    }

                # Bearish breakout
                elif (current['Close'] < range_low - breakout_threshold and
                      current['Low'] < range_low and
                      prev['Close'] >= range_low):

                    breakout_strength = ((range_low - current['Close']) / current_atr) * 100
                    price_change = ((current['Close'] - prev['Close']) / prev['Close']) * 100
                    stop_loss = range_high
                    risk = stop_loss - current['Close']
                    target = current['Close'] - (risk * 2)

                    result = {
                        'Symbol': symbol.replace('.NS', ''),
                        'Breakout_Type': 'Bearish Range Breakout',
                        'Breakout_Strength': round(breakout_strength, 2),
                        'Current_Price': round(current['Close'], 2),
                        'Range_High': round(range_high, 2),
                        'Range_Low': round(range_low, 2),
                        'Range_Size': round(range_size, 2),
                        'Price_Change_%': round(price_change, 2),
                        'Volume_Ratio': round(vol_ratio, 2),
                        'Stop_Loss': round(stop_loss, 2),
                        'Target': round(target, 2),
                        'Risk_Reward': '1:2',
                        'ATR': round(current_atr, 2),
                        'Scan_Time': datetime.now(self.ist).strftime('%H:%M:%S')
                    }

                if result:
                    results.append(result)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        return pd.DataFrame(results)
