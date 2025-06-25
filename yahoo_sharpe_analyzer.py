#!/usr/bin/env python3
"""
Yahoo Finance Sharpe Ratio Reverse Engineering Tool

This script helps identify the exact parameters Yahoo Finance uses for Sharpe ratio calculations
by comparing your calculated ratios with Yahoo's actual reported ratios.

Usage:
python yahoo_sharpe_analyzer.py <price_data.json> --yahoo-1y <value> --yahoo-3y <value> --yahoo-5y <value>

Or modify the YAHOO_SHARPE_RATIOS dictionary in the script with the actual values.
"""

import json
import sys
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict

# You can modify these values or pass them via command line
YAHOO_SHARPE_RATIOS = {
    '1y': None,  # Replace with actual Yahoo 1Y Sharpe ratio
    '3y': None,  # Replace with actual Yahoo 3Y Sharpe ratio
    '5y': None,  # Replace with actual Yahoo 5Y Sharpe ratio
}

def load_yahoo_finance_data(filename: str) -> List[Tuple[int, float]]:
    """Load and parse Yahoo Finance JSON data, filtering null values."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Navigate through the schema structure
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    
    # Get adjusted close prices
    if 'adjclose' in result['indicators'] and result['indicators']['adjclose']:
        prices = result['indicators']['adjclose'][0]['adjclose']
    else:
        prices = result['indicators']['quote'][0]['close']
    
    # Filter out None/null values
    valid_data = []
    for ts, price in zip(timestamps, prices):
        if price is not None and ts is not None:
            valid_data.append((ts, price))
    
    # Sort by timestamp
    valid_data.sort(key=lambda x: x[0])
    return valid_data

def calculate_returns(price_data: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Calculate daily returns from price data."""
    if len(price_data) < 2:
        return []
    
    returns = []
    for i in range(1, len(price_data)):
        prev_price = price_data[i-1][1]
        curr_price = price_data[i][1]
        
        if prev_price > 0:
            daily_return = (curr_price - prev_price) / prev_price
            returns.append((price_data[i][0], daily_return))
    
    return returns

def filter_data_by_period(data: List[Tuple[int, float]], years: float) -> List[Tuple[int, float]]:
    """Filter data to include only the specified number of years from the most recent date."""
    if not data:
        return []
    
    latest_timestamp = data[-1][0]
    cutoff_timestamp = latest_timestamp - (years * 365.25 * 24 * 3600)
    
    filtered_data = [(ts, value) for ts, value in data if ts >= cutoff_timestamp]
    return filtered_data

def calculate_sharpe_ratio_variants(returns: List[Tuple[int, float]], 
                                  risk_free_rate: float = 0.02,
                                  trading_days: int = 252,
                                  use_population_std: bool = False,
                                  return_details: bool = False) -> Optional[float]:
    """
    Calculate Sharpe ratio with various parameter options.
    
    Args:
        returns: List of (timestamp, daily_return) tuples
        risk_free_rate: Annual risk-free rate
        trading_days: Trading days per year for annualization
        use_population_std: Use population std (ddof=0) instead of sample std (ddof=1)
        return_details: If True, return dict with detailed calculations
    """
    if len(returns) < 30:
        return None
    
    return_values = [r[1] for r in returns]
    
    # Calculate statistics
    mean_return = np.mean(return_values)
    if use_population_std:
        std_return = np.std(return_values, ddof=0)  # Population standard deviation
    else:
        std_return = np.std(return_values, ddof=1)  # Sample standard deviation
    
    if std_return == 0:
        return None
    
    # Annualize
    annualized_return = mean_return * trading_days
    annualized_volatility = std_return * np.sqrt(trading_days)
    
    # Calculate Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    if return_details:
        return {
            'sharpe_ratio': sharpe_ratio,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'mean_daily_return': mean_return,
            'daily_volatility': std_return,
            'num_observations': len(return_values),
            'risk_free_rate': risk_free_rate,
            'trading_days': trading_days
        }
    
    return sharpe_ratio

def reverse_engineer_risk_free_rate(returns: List[Tuple[int, float]], 
                                  target_sharpe: float,
                                  trading_days: int = 252,
                                  use_population_std: bool = False) -> Optional[float]:
    """
    Reverse engineer the risk-free rate that would produce the target Sharpe ratio.
    
    Sharpe = (Return - RFR) / Volatility
    Therefore: RFR = Return - (Sharpe * Volatility)
    """
    if len(returns) < 30:
        return None
    
    return_values = [r[1] for r in returns]
    
    mean_return = np.mean(return_values)
    if use_population_std:
        std_return = np.std(return_values, ddof=0)
    else:
        std_return = np.std(return_values, ddof=1)
    
    if std_return == 0:
        return None
    
    # Annualize
    annualized_return = mean_return * trading_days
    annualized_volatility = std_return * np.sqrt(trading_days)
    
    # Reverse calculate risk-free rate
    implied_rfr = annualized_return - (target_sharpe * annualized_volatility)
    
    return implied_rfr

def test_parameter_combinations(returns_data: Dict[str, List[Tuple[int, float]]], 
                              yahoo_ratios: Dict[str, float]) -> None:
    """Test various parameter combinations to match Yahoo's Sharpe ratios."""
    
    # Parameter combinations to test
    trading_days_options = [252, 250, 260, 365]  # Different trading day assumptions
    std_options = [False, True]  # Sample vs Population standard deviation
    risk_free_rates = [0.0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    
    print("\n" + "="*80)
    print("PARAMETER COMBINATION TESTING")
    print("="*80)
    
    best_matches = {}
    
    for period in ['1y', '3y', '5y']:
        if yahoo_ratios[period] is None:
            print(f"\nSkipping {period} - no Yahoo ratio provided")
            continue
            
        if period not in returns_data:
            print(f"\nSkipping {period} - no return data available")
            continue
        
        print(f"\n{period.upper()} SHARPE RATIO ANALYSIS")
        print("-" * 40)
        print(f"Target Yahoo Sharpe Ratio: {yahoo_ratios[period]:.4f}")
        
        best_match = None
        best_diff = float('inf')
        
        results = []
        
        for trading_days in trading_days_options:
            for use_pop_std in std_options:
                for rfr in risk_free_rates:
                    calculated_sharpe = calculate_sharpe_ratio_variants(
                        returns_data[period], 
                        risk_free_rate=rfr,
                        trading_days=trading_days,
                        use_population_std=use_pop_std
                    )
                    
                    if calculated_sharpe is not None:
                        diff = abs(calculated_sharpe - yahoo_ratios[period])
                        
                        results.append({
                            'trading_days': trading_days,
                            'use_pop_std': use_pop_std,
                            'risk_free_rate': rfr,
                            'calculated_sharpe': calculated_sharpe,
                            'difference': diff
                        })
                        
                        if diff < best_diff:
                            best_diff = diff
                            best_match = results[-1].copy()
        
        # Sort results by difference and show top 5 matches
        results.sort(key=lambda x: x['difference'])
        
        print(f"\nTop 5 closest matches:")
        print(f"{'Rank':<4} {'Trading Days':<12} {'Std Type':<10} {'Risk-Free Rate':<15} {'Calculated':<12} {'Difference':<12}")
        print("-" * 75)
        
        for i, result in enumerate(results[:5], 1):
            std_type = "Population" if result['use_pop_std'] else "Sample"
            print(f"{i:<4} {result['trading_days']:<12} {std_type:<10} "
                  f"{result['risk_free_rate']:<15.3%} {result['calculated_sharpe']:<12.4f} "
                  f"{result['difference']:<12.6f}")
        
        best_matches[period] = best_match
        
        # Also show reverse-engineered risk-free rate
        for trading_days in [252, 250]:
            for use_pop_std in [False, True]:
                implied_rfr = reverse_engineer_risk_free_rate(
                    returns_data[period],
                    yahoo_ratios[period],
                    trading_days=trading_days,
                    use_population_std=use_pop_std
                )
                
                if implied_rfr is not None:
                    std_type = "Population" if use_pop_std else "Sample"
                    print(f"\nImplied RFR ({trading_days} days, {std_type} std): {implied_rfr:.4%}")
    
    # Summary of best matches
    print("\n" + "="*80)
    print("SUMMARY OF BEST PARAMETER MATCHES")
    print("="*80)
    
    for period, match in best_matches.items():
        if match:
            std_type = "Population" if match['use_pop_std'] else "Sample"
            print(f"\n{period.upper()} Best Match:")
            print(f"  Trading Days: {match['trading_days']}")
            print(f"  Standard Deviation: {std_type}")
            print(f"  Risk-Free Rate: {match['risk_free_rate']:.3%}")
            print(f"  Calculated Sharpe: {match['calculated_sharpe']:.4f}")
            print(f"  Yahoo Sharpe: {yahoo_ratios[period]:.4f}")
            print(f"  Difference: {match['difference']:.6f}")

def detailed_calculation_analysis(returns_data: Dict[str, List[Tuple[int, float]]], 
                                yahoo_ratios: Dict[str, float]) -> None:
    """Provide detailed analysis of calculations with different parameters."""
    
    print("\n" + "="*80)
    print("DETAILED CALCULATION ANALYSIS")
    print("="*80)
    
    # Use most common parameters for detailed analysis
    standard_params = {
        'risk_free_rate': 0.02,
        'trading_days': 252,
        'use_population_std': False
    }
    
    for period in ['1y', '3y', '5y']:
        if period not in returns_data or yahoo_ratios[period] is None:
            continue
        
        print(f"\n{period.upper()} DETAILED ANALYSIS")
        print("-" * 50)
        
        # Calculate with standard parameters
        details = calculate_sharpe_ratio_variants(
            returns_data[period],
            return_details=True,
            **standard_params
        )
        
        if details:
            print(f"Data Points: {details['num_observations']}")
            print(f"Mean Daily Return: {details['mean_daily_return']:.6f} ({details['mean_daily_return']*100:.4f}%)")
            print(f"Daily Volatility: {details['daily_volatility']:.6f} ({details['daily_volatility']*100:.4f}%)")
            print(f"Annualized Return: {details['annualized_return']:.4%}")
            print(f"Annualized Volatility: {details['annualized_volatility']:.4%}")
            print(f"Risk-Free Rate Used: {details['risk_free_rate']:.4%}")
            print(f"Trading Days Used: {details['trading_days']}")
            print(f"Calculated Sharpe Ratio: {details['sharpe_ratio']:.4f}")
            print(f"Yahoo Sharpe Ratio: {yahoo_ratios[period]:.4f}")
            print(f"Difference: {abs(details['sharpe_ratio'] - yahoo_ratios[period]):.4f}")

def main():
    parser = argparse.ArgumentParser(description='Reverse engineer Yahoo Finance Sharpe ratio calculations')
    parser.add_argument('filename', help='Yahoo Finance JSON file with price data')
    parser.add_argument('--yahoo-1y', type=float, help='Yahoo Finance 1Y Sharpe ratio')
    parser.add_argument('--yahoo-3y', type=float, help='Yahoo Finance 3Y Sharpe ratio')
    parser.add_argument('--yahoo-5y', type=float, help='Yahoo Finance 5Y Sharpe ratio')
    
    args = parser.parse_args()
    
    # Update Yahoo ratios from command line or use defaults from script
    yahoo_ratios = YAHOO_SHARPE_RATIOS.copy()
    if args.yahoo_1y is not None:
        yahoo_ratios['1y'] = args.yahoo_1y
    if args.yahoo_3y is not None:
        yahoo_ratios['3y'] = args.yahoo_3y
    if args.yahoo_5y is not None:
        yahoo_ratios['5y'] = args.yahoo_5y
    
    # Check if we have any Yahoo ratios to work with
    if all(v is None for v in yahoo_ratios.values()):
        print("Error: No Yahoo Finance Sharpe ratios provided.")
        print("Either modify YAHOO_SHARPE_RATIOS in the script or use command line arguments:")
        print("  --yahoo-1y <value> --yahoo-3y <value> --yahoo-5y <value>")
        sys.exit(1)
    
    try:
        # Load price data
        print(f"Loading price data from {args.filename}...")
        price_data = load_yahoo_finance_data(args.filename)
        print(f"Loaded {len(price_data)} valid price data points")
        
        # Calculate returns
        all_returns = calculate_returns(price_data)
        print(f"Calculated {len(all_returns)} daily returns")
        
        if len(all_returns) < 30:
            print("Error: Insufficient return data for analysis")
            sys.exit(1)
        
        # Prepare return data for different periods
        returns_data = {}
        periods = [(1, '1y'), (3, '3y'), (5, '5y')]
        
        for years, period_key in periods:
            period_returns = filter_data_by_period(all_returns, years)
            if len(period_returns) >= 30:  # Minimum data requirement
                returns_data[period_key] = period_returns
                print(f"{period_key}: {len(period_returns)} return observations")
        
        if not returns_data:
            print("Error: No periods have sufficient data for analysis")
            sys.exit(1)
        
        # Run analysis
        test_parameter_combinations(returns_data, yahoo_ratios)
        detailed_calculation_analysis(returns_data, yahoo_ratios)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("\nKey findings to look for:")
        print("1. Consistent risk-free rate across all periods")
        print("2. Consistent trading days assumption (likely 252 or 250)")
        print("3. Standard deviation calculation method (sample vs population)")
        print("4. Very small differences (<0.001) indicate likely parameter match")
        
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{args.filename}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()