#!/usr/bin/env python3
"""
Multi-Format Sharpe Ratio Calculator

Calculates 1Y, 3Y, and 5Y Sharpe ratios from Yahoo Finance or Morningstar JSON data.
Automatically detects the file format and parses accordingly.
Allows specifying a custom risk-free rate as a percentage (default 2%).
Properly handles null values by dropping those data points.
"""

import json
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np

def detect_json_format(data):
    """
    Detect whether the JSON is Yahoo Finance or Morningstar format.
    
    Returns:
        'yahoo' for Yahoo Finance format
        'morningstar' for Morningstar format
        'unknown' if format cannot be determined
    """
    # Check for Yahoo Finance format
    if 'chart' in data and 'result' in data['chart']:
        return 'yahoo'
    
    # Check for Morningstar format
    if 'TimeSeries' in data and 'Security' in data['TimeSeries']:
        return 'morningstar'
    
    return 'unknown'

def load_yahoo_finance_data(data):
    """Load and parse Yahoo Finance JSON data, dropping null adjclose values."""
    # Navigate through the schema structure
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    
    print(f"Yahoo Finance format detected")
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
            if invalid_timestamp_count <= 5:  # Only print first 5 invalid timestamps
                print(f"    Invalid timestamp at index {i}: {ts}")
            continue
            
        valid_data.append((ts, price))
    
    print(f"  Null price data points dropped: {null_count}")
    print(f"  Invalid timestamp data points dropped: {invalid_timestamp_count}")
    print(f"  Valid data points retained: {len(valid_data)}")
    
    return valid_data

def load_morningstar_data(data):
    """Load and parse Morningstar JSON data, dropping null values."""
    # Navigate through the schema structure
    securities = data['TimeSeries']['Security']
    
    if not securities:
        raise ValueError("No securities found in Morningstar data")
    
    # Use the first security (assuming single security data)
    security = securities[0]
    history_details = security.get('HistoryDetail', [])
    
    print(f"Morningstar format detected")
    print(f"  Total data points: {len(history_details)}")
    
    valid_data = []
    null_count = 0
    invalid_date_count = 0
    
    for i, detail in enumerate(history_details):
        # Extract data
        end_date = detail.get('EndDate')
        value = detail.get('Value')
        
        # Check for valid price/value
        if value is None:
            null_count += 1
            continue
            
        # Try to convert value to float
        try:
            price = float(value)
        except (ValueError, TypeError):
            null_count += 1
            continue
        
        # Check for valid date
        if end_date is None:
            invalid_date_count += 1
            continue
            
        # Convert date string to timestamp
        try:
            date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            timestamp = int(date_obj.timestamp())
        except ValueError:
            invalid_date_count += 1
            if invalid_date_count <= 5:  # Only print first 5 invalid dates
                print(f"    Invalid date at index {i}: {end_date}")
            continue
        
        valid_data.append((timestamp, price))
    
    print(f"  Null/invalid price data points dropped: {null_count}")
    print(f"  Invalid date data points dropped: {invalid_date_count}")
    print(f"  Valid data points retained: {len(valid_data)}")
    
    return valid_data

def load_financial_data(filename):
    """
    Load financial data from either Yahoo Finance or Morningstar JSON format.
    Automatically detects the format and parses accordingly.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Detect format
    format_type = detect_json_format(data)
    
    if format_type == 'yahoo':
        valid_data = load_yahoo_finance_data(data)
    elif format_type == 'morningstar':
        valid_data = load_morningstar_data(data)
    else:
        raise ValueError(f"Unknown JSON format. Expected Yahoo Finance or Morningstar format.")
    
    if not valid_data:
        raise ValueError("No valid price data found after filtering null values and invalid timestamps/dates")
    
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
        risk_free_rate: Annual risk-free rate (as decimal, e.g., 0.02 for 2%)
    
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate Sharpe ratios from Yahoo Finance or Morningstar JSON data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sharpe_calculator.py data.json
  python sharpe_calculator.py data.json --risk-free-rate 3.5
  python sharpe_calculator.py data.json -r 1.75

The risk-free rate should be provided as a percentage (e.g., 2.5 for 2.5%).
Default risk-free rate is 2.0%.
        """
    )
    
    parser.add_argument('filename', 
                       help='JSON file containing price data (Yahoo Finance or Morningstar format)')
    
    parser.add_argument('-r', '--risk-free-rate', 
                       type=float, 
                       default=2.0,
                       help='Risk-free rate as a percentage (default: 2.0%%)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    filename = args.filename
    risk_free_rate_percent = args.risk_free_rate
    
    # Convert percentage to decimal
    risk_free_rate = risk_free_rate_percent / 100.0
    
    # Validate risk-free rate
    if risk_free_rate < 0 or risk_free_rate > 1:
        print(f"Warning: Risk-free rate of {risk_free_rate_percent}% seems unusual.")
        print("Expected range is typically 0% to 20%. Proceeding anyway...")
    
    try:
        # Load price data (automatically detects format and filters null values)
        print(f"Loading data from {filename}...")
        print(f"Using risk-free rate: {risk_free_rate_percent}% ({risk_free_rate:.4f})")
        price_data = load_financial_data(filename)
        
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
        print(f"Risk-free rate: {risk_free_rate_percent}%")
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
            sharpe = calculate_sharpe_ratio(period_returns, risk_free_rate)
            
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
        print("      Null values and invalid timestamps/dates have been filtered out")
        
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