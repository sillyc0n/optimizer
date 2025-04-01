import numpy as np
from scipy.optimize import minimize
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Portfolio Optimizer")
parser.add_argument("funds_file", help="Path to funds data CSV file")
parser.add_argument("covariance_file", help="Path to covariance matrix CSV file")
parser.add_argument("--sedol_list", nargs='+', help="List of SEDOLs to include")

args = parser.parse_args()

fd = pd.read_csv(args.funds_file)
print(f"Total number of funds: {len(fd)}")

cm = pd.read_csv(args.covariance_file, index_col=0)
print(f"Correlation matrix dimensions: {len(cm)} rows, {len(cm.columns)} columns")

# filter fd to only contain sedol_list
if args.sedol_list:
    fd = fd.dropna(subset=['yahoo_symbol'])
    fd = fd[fd['sedol'].isin(args.sedol_list)]
    print(f"Filtered fd to include only specified SEDOLs: {len(fd)}")

    unique_symbols = fd['yahoo_symbol'].unique()

    print(f"Unique symbols: {unique_symbols}")
    cm = cm.loc[unique_symbols, unique_symbols]
    print(f"Filtered cm to include only specified SEDOLs: {len(cm)} x {len(cm.columns)}")

unique_symbols = fd['yahoo_symbol'].unique()
if len(unique_symbols) == len(fd):
    print("All entries in fd are unique on yahoo_symbol")
else:
    print(f"Duplicate yahoo_symbol(s) in fd. Total unique symbols: {len(unique_symbols)}")
        
    duplicates = fd[fd['yahoo_symbol'].isin(unique_symbols)].groupby('yahoo_symbol').size().reset_index(name='count')
    duplicates = duplicates[duplicates['count'] > 1]
    print(duplicates)

# remove duplicates
fd = fd.drop_duplicates(subset='yahoo_symbol', keep='first')
print(f"After removing duplicates: len(fd) = {len(fd)}")

# filter fd to contain only rows with entries in cm
print(f"len(fd): {len(fd)}")
fd = fd[fd['yahoo_symbol'].isin(cm.index)].reset_index(drop=True)
print(f"len(fd) isin(cm.index): {len(fd)}")

symbols_not_in_cm = fd['yahoo_symbol'][~fd['yahoo_symbol'].isin(cm.index)]
if len(symbols_not_in_cm) > 0:
    print(f"Symbols in fd not found in cm: {symbols_not_in_cm.tolist()}")
else:
    print("All symbols in fd are present in cm")

cm_symbols = set(cm.index)
fd_symbols = set(fd['yahoo_symbol'])

missing_symbols = cm_symbols - fd_symbols

if missing_symbols:
    print(f"Symbols in cm not found in fd: {list(missing_symbols)}")
else:
    print("All symbols in cm are present in fd")

print(f"I have {len(fd)} funds and the Correlation Matrix is {len(cm)} x {len(cm.columns)}")

print(f"Crunching some numbers. This will take a moment!")
class PortfolioOptimizer:
    def __init__(self, funds_data, covariance_matrix):
        """
        Initialize optimizer with fund data and covariance matrix
        
        Parameters:
        funds_data: DataFrame with columns [sedol, distribution_yield, sharpeRatios_oneYear, annual_charge]
        covariance_matrix: DataFrame with sedols as index/columns containing covariances
        """
        self.funds_data = funds_data
        self.covariance_matrix = covariance_matrix        
        self.n_funds = len(funds_data)
        
    def objective_function(self, weights):
        """
        Multi-objective function to optimize:
        - Maximize distribution yield
        - Maximize Sharpe ratio
        - Minimize annual charge
        - Minimize portfolio variance
        
        Returns negative value since scipy.optimize minimizes
        """
        # Calculate weighted metrics
        port_dist_yield = np.sum(weights * self.funds_data['distribution_yield'])
        port_sharpe = np.sum(weights * self.funds_data['fidelity_sharpeRatios_oneYear'])
        port_charge = np.sum(weights * self.funds_data['annual_charge'])
        
        # Calculate portfolio variance
        port_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        
        # Combine objectives with weights for importance
        # Negative because we want to maximize yield and sharpe
        objective = -(
            0.3 * port_dist_yield +  # 30% weight on yield
            0.3 * port_sharpe +      # 30% weight on sharpe
            -0.2 * port_charge +     # 20% weight on minimizing charges
            -0.2 * port_variance     # 20% weight on minimizing variance
        )
        
        return objective
    
    def portfolio_risk(self, weights):
        """Calculate portfolio risk (standard deviation)"""
        return np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
    
    def optimize_portfolio(self, min_weight=0.05, max_weight=0.3):
        """
        Optimize the portfolio allocation
        
        Parameters:
        min_weight: minimum weight for any fund (default 5%)
        max_weight: maximum weight for any fund (default 30%)
        
        Returns:
        optimal_weights: array of optimal weights
        portfolio_metrics: dict of resulting portfolio metrics
        """
        # Constraints
        constraints = [
            {
                'type': 'eq', 
                'fun': lambda x: np.sum(x) - 1
            },  # weights sum to 1
        ]
        # Bounds for each weight
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_funds))       # ((0.05, 0.3), ... repeated n_funds times )
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/self.n_funds] * self.n_funds)                 # [ 1/n_funds repeated n_funds times ]
        
        # Optimize - Minimize a scalar function of one or more variables using Sequential Least Squares Programming (SLSQP)
        result = minimize(                          
            self.objective_function,                                                # minimize objective function
            initial_weights,                                                        # first intuition
            method='SLSQP',
            bounds=bounds,                                                          # Bounds constraint on the variables.
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        # Calculate resulting portfolio metrics
        portfolio_metrics = {
            'distribution_yield': np.sum(optimal_weights * self.funds_data['distribution_yield']),
            'sharpe_ratio': np.sum(optimal_weights * self.funds_data['fidelity_sharpeRatios_oneYear']),
            'annual_charge': np.sum(optimal_weights * self.funds_data['annual_charge']),
            'portfolio_risk': self.portfolio_risk(optimal_weights),
            'success': result.success,
            'optimization_message': result.message
        }
        
        return optimal_weights, portfolio_metrics

    def generate_efficient_frontier(self, n_points=50):
        """
        Generate efficient frontier by varying target returns
        Returns points for plotting efficient frontier
        """        
        min_yield = np.min(self.funds_data['distribution_yield'])
        max_yield = np.max(self.funds_data['distribution_yield'])
        
        print(f"generate efficient frontier - min_yield: {min_yield}, max_yield: {max_yield}")

        target_yields = np.linspace(min_yield, max_yield, n_points)                                         # an array of target yields
        frontier_points = []
        
        for target in target_yields:
            # Add target yield constraint
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},                     # weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(x * self.funds_data['distribution_yield']) - target} # overall portfolio's weighted average yield equals to target
            ]
            
            # Optimize for minimum variance at this target yield
            result = minimize(
                lambda w: self.portfolio_risk(w),                                   # objective function
                np.array([1/self.n_funds] * self.n_funds),                          # initial intuition
                method='SLSQP',
                bounds=tuple((0, 1) for _ in range(self.n_funds)),                  # funds allocation between 0 and 1
                constraints=constraints
            )
            
            if result.success:
                risk = self.portfolio_risk(result.x)
                frontier_points.append({
                    'risk': risk,
                    'yield': target,
                    'weights': result.x,                   
                })
        
        return frontier_points

    def plot_efficient_frontier(self, frontier):
        print(self.funds_data)
        risks = [point['risk'] for point in frontier]
        yields = [point['yield'] for point in frontier]
        weights = [point['weights'] for point in frontier]
    
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, (risk, yield_, weight) in enumerate(zip(risks, yields, weights)):            
            weights_percentage = [f'{float(w)*100:2.2f}%' for w in weight]
            print(f"yield: {yield_: .3f}, weights: {weights_percentage}, risk: {risk}")

        #    sedols = self.funds_data.iloc[np.argmax(weight), 0]
        #    allocation = np.round(np.sum(weight), 2)
        #    description = f'SEDOLs: {sedols}\nAllocation: {allocation}'
        #    print(description)
            
        #def hover(event):
        #    if event.inaxes == ax:
        #        cont, labels = plt.get_legend_handles_labels()
        #        return [plt.annotate(description, xy=(risk, yield_), xytext=(10,10),
        #                              textcoords='offset points', ha='left', va='bottom',
        #                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        #                for i, risk, yield_ in zip(range(len(risks)), risks, yields) if event.xdata == risk and event.ydata == yield_]
        #                              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        #fig.canvas.mpl_connect('motion_notify_event', hover)
        
        plt.scatter(risks, yields, alpha=0.5)
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.title('Efficient Frontier')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

# Example usage:
'''
# Create sample data
funds_data = pd.DataFrame({
    'sedol': ['ABC123', 'DEF456', 'GHI789'],
    'distribution_yield': [0.04, 0.03, 0.05],
    'sharpeRatios_oneYear': [1.2, 0.8, 1.5],
    'annual_charge': [0.0075, 0.0050, 0.0100]
})

# Sample covariance matrix
covariance_matrix = np.array([
    [0.04, 0.02, 0.01],
    [0.02, 0.05, 0.02],
    [0.01, 0.02, 0.06]
])
'''
# Initialize optimizer
optimizer = PortfolioOptimizer(fd, cm)

# Optimize portfolio
optimal_weights, metrics = optimizer.optimize_portfolio()
print(f"optimal_weights: {optimal_weights}")
print(f"metrics: {metrics}")
# Generate efficient frontier
frontier = optimizer.generate_efficient_frontier()

# print (f"Efficient frontier: {frontier}")

optimizer.plot_efficient_frontier(frontier)
