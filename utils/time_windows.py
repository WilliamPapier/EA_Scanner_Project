"""
Time window utilities for trading system.
Implements hot zone detection for high-probability time windows.
"""

from datetime import datetime


def is_in_20min_hot_zone(time_str):
    """
    Check if given time is in 20-minute hot zones.
    Hot zones are defined as :50-:10 and :20-:40 windows within each hour.
    
    Args:
        time_str (str): Time string in format that can be parsed by datetime
                       (e.g., "2023-08-14T09:05:00Z", "09:05:00", "09:05")
    
    Returns:
        bool: True if time is in hot zone, False otherwise
    """
    try:
        # Handle different time formats
        if 'T' in time_str:
            # ISO format with date
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        elif ':' in time_str and len(time_str.split(':')) >= 2:
            # Time format like "09:05:00" or "09:05"
            time_parts = time_str.split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            dt = datetime(2023, 1, 1, hour, minute)  # Use dummy date
        else:
            return False
            
        minute = dt.minute
        
        # Hot zone 1: :50-:10 (crosses hour boundary)
        # This means minutes 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        hot_zone_1 = minute >= 50 or minute <= 10
        
        # Hot zone 2: :20-:40
        # This means minutes 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
        hot_zone_2 = 20 <= minute <= 40
        
        return hot_zone_1 or hot_zone_2
        
    except (ValueError, IndexError, AttributeError):
        return False


def get_time_zone_risk_multiplier(time_str):
    """
    Get risk multiplier based on time zone.
    Higher multiplier for hot zones, lower for regular times.
    
    Args:
        time_str (str): Time string
        
    Returns:
        float: Risk multiplier (1.2 for hot zone, 0.8 for regular time)
    """
    if is_in_20min_hot_zone(time_str):
        return 1.2  # Higher risk/reward in hot zones
    else:
        return 0.8  # Lower risk outside hot zones