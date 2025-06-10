#!/usr/bin/env python3
"""
Yahoo Finance Sharpe Ratio Calculator

Calculates 1Y, 3Y, and 5Y Sharpe ratios from Yahoo Finance JSON data.
The script assumes a risk-free rate of 2% annually (adjustable).
Properly handles null adjclose values by dropping those data points.
"""

import json
import sys
from datetime import datetime, timedelta
import numpy as np

def load_yahoo_finance_data(filename):
    """Load and parse Yahoo Finance JSON data, dropping null adjclose values."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Navigate through the schema structure
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    
    # Debug: Print timestamp info
    print(f"Raw data analysis:")
    print(f"  Total data points: {len(timestamps)}")
    print(f"  First few timestamps: {timestamps[:5]}")
    print(f"  Last few timestamps: {timestamps[-5:]}")
    
    # Get adjusted close prices (preferred for calculations)
    if 'adjclose' in result['indicators'] and result['indicators']['adjclose']:
        prices = result['indicators']['adjclose'][0]['adjclose']
        print("  Using adjusted close prices")
    else:
        # Fallback to regular close prices
        prices = result['indicators']['quote'][0]['close']
        print("  Using regular close prices (adjclose not available)")
    
    # Filter out None/null values and invalid timestamps
    valid_data = []
    null_count = 0
    invalid_timestamp_count = 0
    
    # Define reasonable timestamp bounds (Unix timestamps)
    # Jan 1, 2000 to Jan 1, 2030
    min_timestamp = 946684800   # 2000-01-01
    max_timestamp = 1893456000  # 2030-01-01
    
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        # Check for valid price
        if price is None:
            null_count += 1
            continue
            
        # Check for valid timestamp
        if ts is None or ts < min_timestamp or ts > max_timestamp:
            invalid_timestamp_count += 1
            print(f"    Invalid timestamp at index {i}: {ts}")
            continue
            
        valid_data.append((ts, price))
    
    print(f"  Null price data points dropped: {null_count}")
    print(f"  Invalid timestamp data points dropped: {invalid_timestamp_count}")
    print(f"  Valid data points retained: {len(valid_data)}")
    
    if not valid_data:
        raise ValueError("No valid price data found after filtering null values and invalid timestamps")
    
    # Sort by timestamp (should already be sorted, but ensure it)
    valid_data.sort(key=lambda x: x[0])
    
    # Debug: Print converted timestamp info
    print(f"  Timestamp range after filtering:")
    try:
        first_date = datetime.fromtimestamp(valid_data[0][0]).strftime('%Y-%m-%d')
        last_date = datetime.fromtimestamp(valid_data[-1][0]).strftime('%Y-%m-%d')
        print(f"    First: {valid_data[0][0]} -> {first_date}")
        print(f"    Last: {valid_data[-1][0]} -> {last_date}")
        
        time_span_days = (valid_data[-1][0] - valid_data[0][0]) / (24 * 3600)
        print(f"    Time span: {time_span_days:.1f} days ({time_span_days/365.25:.1f} years)")
    except (ValueError, OSError) as e:
        print(f"    Error converting timestamps: {e}")
        print(f"    Raw range: {valid_data[0][0]} to {valid_data[-1][0]}")
        
        # If timestamp conversion still fails, let's examine the data more closely
        print(f"    Debugging timestamp issues:")
        print(f"      Min timestamp in data: {min([ts for ts, _ in valid_data])}")
        print(f"      Max timestamp in data: {max([ts for ts, _ in valid_data])}")
        
        # Try to identify problematic timestamps
        problematic_timestamps = []
        for ts, _ in valid_data[:10]:  # Check first 10
            try:
                datetime.fromtimestamp(ts)
            except (ValueError, OSError):
                problematic_timestamps.append(ts)
        
        if problematic_timestamps:
            print(f"      Problematic timestamps found: {problematic_timestamps}")
    
    return valid_data

def calculate_returns(price_data):
    """Calculate daily returns from price data (already filtered for null values)."""
    if len(price_data) < 2:
        return []
    
    returns = []
    for i in range(1, len(price_data)):
        prev_price = price_data[i-1][1]
        curr_price = price_data[i][1]
        
        # Additional safety check (shouldn't be needed after filtering, but good practice)
        if prev_price is None or curr_price is None or prev_price <= 0:
            continue
            
        daily_return = (curr_price - prev_price) / prev_price
        returns.append((price_data[i][0], daily_return))
    
    return returns

def filter_data_by_period(data, years):
    """Filter data to include only the specified number of years from the most recent date."""
    if not data:
        return []
    
    latest_timestamp = data[-1][0]
    cutoff_timestamp = latest_timestamp - (years * 365.25 * 24 * 3600)  # Approximate years to seconds
    
    filtered_data = [(ts, value) for ts, value in data if ts >= cutoff_timestamp]
    return filtered_data

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: List of tuples (timestamp, daily_return)
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Sharpe ratio or None if insufficient data
    """
    if len(returns) < 30:  # Need at least 30 days of data
        return None
    
    # Extract just the return values
    return_values = [r[1] for r in returns]
    
    # Additional filtering for extreme outliers that might indicate data issues
    return_values = [r for r in return_values if abs(r) < 1.0]  # Remove >100% daily moves (likely data errors)
    
    if len(return_values) < 30:
        return None
    
    # Calculate statistics
    mean_return = np.mean(return_values)
    std_return = np.std(return_values, ddof=1)  # Sample standard deviation
    
    if std_return == 0:
        return None
    
    # Annualize the returns and volatility
    # Assuming ~252 trading days per year
    trading_days_per_year = 252
    annualized_return = mean_return * trading_days_per_year
    annualized_volatility = std_return * np.sqrt(trading_days_per_year)
    
    # Calculate Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    return sharpe_ratio

def main():
    if len(sys.argv) != 2:
        print("Usage: python sharpe_calculator.py <yahoo_finance_json_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        # Load price data (null values already filtered out)
        print(f"Loading data from {filename}...")
        price_data = load_yahoo_finance_data(filename)
        
        print(f"Proceeding with {len(price_data)} valid data points")
        
        if len(price_data) < 2:
            print("Error: Insufficient valid price data for calculations.")
            sys.exit(1)
        
        # Calculate daily returns
        all_returns = calculate_returns(price_data)
        
        if not all_returns:
            print("Error: Unable to calculate returns from available data.")
            sys.exit(1)
        
        print(f"Calculated {len(all_returns)} daily returns")
        
        # Check available data span and only calculate for feasible periods
        start_ts = price_data[0][0]
        end_ts = price_data[-1][0]
        available_years = (end_ts - start_ts) / (365.25 * 24 * 3600)
        
        # Define periods, but only include those with sufficient data
        all_periods = [
            (1, "1 Year"),
            (3, "3 Year"), 
            (5, "5 Year")
        ]
        
        # Filter periods based on available data (require at least 80% of the period)
        periods = [(years, name) for years, name in all_periods if available_years >= years * 0.8]
        
        if not periods:
            print(f"Error: Insufficient data span ({available_years:.1f} years) for any meaningful Sharpe ratio calculation.")
            sys.exit(1)
        
        print("\n" + "="*50)
        print("SHARPE RATIO ANALYSIS")
        print("="*50)
        print(f"Available data span: {available_years:.1f} years")
        print(f"Calculating Sharpe ratios for: {', '.join([name for _, name in periods])}")
        print("-"*50)
        
        for years, period_name in periods:
            # Filter returns for the specific period
            period_returns = filter_data_by_period(all_returns, years)
            
            if len(period_returns) < 30:
                print(f"{period_name:10}: Insufficient data ({len(period_returns)} days)")
                continue
            
            # Calculate Sharpe ratio
            sharpe = calculate_sharpe_ratio(period_returns)
            
            if sharpe is not None:
                # Get some additional statistics
                return_values = [r[1] for r in period_returns]
                # Filter extreme outliers for statistics (same as in Sharpe calculation)
                clean_returns = [r for r in return_values if abs(r) < 1.0]
                
                annualized_return = np.mean(clean_returns) * 252
                annualized_volatility = np.std(clean_returns, ddof=1) * np.sqrt(252)
                
                print(f"{period_name:10}: {sharpe:.3f}")
                print(f"{'':10}  Ann. Return: {annualized_return*100:6.2f}%")
                print(f"{'':10}  Ann. Volatility: {annualized_volatility*100:6.2f}%")
                print(f"{'':10}  Days: {len(period_returns)} ({len(clean_returns)} after outlier filter)")
                print("-"*50)
            else:
                print(f"{period_name:10}: Unable to calculate (insufficient variance or data)")
        
        print("\nNote: Sharpe ratios > 1.0 are generally considered good")
        print("      Sharpe ratios > 2.0 are considered very good")
        print("      Null adjclose values have been filtered out")
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{filename}'.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Expected key not found in JSON structure: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()