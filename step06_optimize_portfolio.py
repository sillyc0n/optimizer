import numpy as np
from scipy.optimize import minimize
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import mplcursors

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
    # Only keep symbols that exist in cm
    symbols_in_cm = [s for s in unique_symbols if s in cm.index and s in cm.columns]
    missing = set(unique_symbols) - set(symbols_in_cm)
    if missing:
        print(f"Symbols not found in covariance matrix and will be dropped: {missing}")
    # Filter fd and cm to only those symbols
    fd = fd[fd['yahoo_symbol'].isin(symbols_in_cm)]
    cm = cm.loc[symbols_in_cm, symbols_in_cm]
    print(f"Filtered fd and cm to include only symbols present in both. fd: {len(fd)}, cm: {len(cm)} x {len(cm.columns)}")

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

    def objective_function_maximize_sharpe_ratio(self, weights):
        port_return = np.sum(weights * (self.funds_data['distribution_yield'] - self.funds_data['total_expenses']))
        port_risk = self.portfolio_risk(weights)
        risk_free_rate = 0.0525
        sharpe = (port_return - risk_free_rate) / port_risk
        return -sharpe
        
    def objective_function_multi_objective(self, weights):
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
        
        # Add penalty term for unequal weighting
        penalty = np.sum((weights - 1/self.n_funds)**2)

        # Combine objectives with weights for importance
        # Negative because we want to maximize yield and sharpe
        objective = -(
            0 * port_dist_yield +  # % weight on yield
            1 * port_sharpe +      # % weight on sharpe
            -0 * port_charge +     # % weight on minimizing charges
            -0 * port_variance +   # % weight on minimizing variance
            0
            #0.1 * penalty            # New term to maintain diversification
        )
        
        return objective
    
    def portfolio_risk(self, weights):
        """Calculate portfolio risk (standard deviation)"""
        return np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
    
    
    def optimize_portfolio(self, min_weight=0.05, max_weight=0.3):
        """
        Optimize the portfolio allocation balancing the objectives.
        
        Parameters:
        min_weight: minimum weight for any fund (default 5%)
        max_weight: maximum weight for any fund (default 30%)
        
        Returns:
        optimal_weights: array of optimal weights
        portfolio_metrics: dict of resulting portfolio metrics
        """
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},         # weights sum to 1
            #{'type': 'ineq', 'fun': lambda x: min_weight - x}, 
            #{'type': 'ineq', 'fun': lambda x: x - max_weight}
        ]
        # Bounds for each weight
        #bounds = tuple((min_weight, max_weight) for _ in range(self.n_funds))       # ((0.05, 0.3), ... repeated n_funds times )
        bounds=tuple((0, 1) for _ in range(self.n_funds))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/self.n_funds] * self.n_funds)                 # [ 1/n_funds repeated n_funds times ]
        
        # Optimize - Minimize a scalar function of one or more variables using Sequential Least Squares Programming (SLSQP)
        result = minimize(                          
            #self.objective_function_multi_objective,                                # minimize objective function
            self.objective_function_maximize_sharpe_ratio,                           # maximize Sharpe ratio
            initial_weights,                                                        # first intuition
            method='SLSQP',
            bounds=bounds,                                                          # Bounds constraint on the variables.
            constraints=constraints
        )

        optimal_weights = result.x

        #print(result)

        optimal_weights_df = pd.DataFrame({
            'sedol': self.funds_data['sedol'],
            'fund_name': self.funds_data['fund_name'],
            'distribution_yield': self.funds_data['distribution_yield'],
            'weight': optimal_weights
        })

        # Calculate resulting portfolio metrics
        portfolio_metrics = {
            'distribution_yield': np.sum(optimal_weights * self.funds_data['distribution_yield']),
            'sharpe_ratio': np.sum(optimal_weights * self.funds_data['fidelity_sharpeRatios_oneYear']),
            'annual_charge': np.sum(optimal_weights * self.funds_data['annual_charge']),
            'portfolio_risk': self.portfolio_risk(optimal_weights),
            'success': result.success,
            'optimization_message': result.message
        }
        return optimal_weights_df, portfolio_metrics

    def generate_efficient_frontier(self, n_points=50):
        """
        Generate efficient frontier by varying target returns
        Returns points for plotting efficient frontier
        """
        #min_yield = np.min(self.funds_data['distribution_yield'])
        #max_yield = np.max(self.funds_data['distribution_yield'])

        #print(f"generate efficient frontier - min_yield: {min_yield}, max_yield: {max_yield}")

        min_net_yield = np.min(self.funds_data['distribution_yield'] - self.funds_data['total_expenses'])
        max_net_yield = np.max(self.funds_data['distribution_yield'] - self.funds_data['total_expenses'])

        print(f"generate efficient frontier - min_net_yield: {min_net_yield}, max_net_yield: {max_net_yield}")

        target_yields = np.linspace(min_net_yield, max_net_yield, n_points)                                         # an array of target yields        
        frontier_points = []

        for target in target_yields:
            # Add target yield constraint
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},                     # weights sum to 1
                #{'type': 'eq', 'fun': lambda x: np.sum(x * self.funds_data['distribution_yield']) - target} # overall portfolio's weighted average yield equals to target
                {'type': 'eq', 'fun': lambda x: np.sum(x * (self.funds_data['distribution_yield'] - self.funds_data['total_expenses'])) - target} # overall portfolio's weighted average yield equals to target
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
                    #'yield': target,
                    'net_yield': target,
                    'weights': result.x,
                })

        return frontier_points

    def plot_efficient_frontier(self, frontier, ax=None):
        risks = [point['risk'] for point in frontier]                
        yields = [point['net_yield'] for point in frontier]
        weights = [point['weights'] for point in frontier]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Plot efficient frontier points
        # Plot efficient frontier points as a scatter for interactivity
        scatter = ax.scatter(risks, yields, color='black', label='Efficient Frontier', zorder=10)

        # Calculate Sharpe ratios for each point
        risk_free_rate = 0.0525  # 5.25%
        sharpe_ratios = [(y - risk_free_rate) / r for y, r in zip(yields, risks)]
        
        # Find tangent portfolio (maximum Sharpe ratio)
        tangent_idx = np.argmax(sharpe_ratios)
        tangent_risk = risks[tangent_idx]
        tangent_yield = yields[tangent_idx]
        
        # Plot tangent portfolio
        ax.scatter(tangent_risk, tangent_yield, color='red', s=100, label='Tangent Portfolio')
        
        # Plot risk-free rate point
        ax.scatter(0, risk_free_rate, color='green', s=100, label='Risk-Free Rate')

        # Draw Capital Market Line with dynamic range
        min_risk = min(risks)
        max_risk = max(risks)
        max_yield = max(yields)
        min_yield = min(yields)

        # Draw Capital Market Line
        slope = (tangent_yield - risk_free_rate) / tangent_risk
        max_x = (1.5 * max_yield - risk_free_rate) / slope

        # Calculate ranges for efficient frontier
        risk_range = max_risk - min_risk
        yield_range = max_yield - min_yield

        # Calculate padding to make efficient frontier occupy 60-80% of the plot
        x_padding = risk_range * 0.15  # 25% padding on each side
        y_padding = yield_range * 0.15  # 25% padding on each side

        # Set axis limits with calculated padding
        ax.set_xlim(min_risk - x_padding, min(max_risk, max_x) + x_padding)
        ax.set_ylim(min_yield - y_padding, max_yield + y_padding)

        # Draw Capital Market Line extending to the left and limited on the right
        # Dynamically adjust right limit based on slope
        # Steeper slopes (higher Sharpe ratios) get more limited
        tangent_yield = yields[tangent_idx]
        slope = (tangent_yield - risk_free_rate) / tangent_risk
        slope_factor = 1.0 / (1.0 + abs(slope))  # Inverse relationship with slope
        #right_limit = tangent_risk * (1.0 + slope_factor * 0.2)  # Max 20% extension for steep slopes
        right_limit = tangent_risk * (1.0 + slope_factor)  # Max 20% extension for steep slopes

        #x = np.linspace(0, max(risks)*1.2, 100)
        x = np.linspace(0, right_limit, 100)  # Start from 0 (risk-free rate)

        y = risk_free_rate + slope * x
        ax.plot(x, y, 'r--', label='Capital Market Line')

        # Add interactive hover tooltips for efficient frontier points
        sedols = list(self.funds_data['sedol'])
        fund_names = list(self.funds_data['fund_name'])
        def tooltip_text(index):
            w = weights[index]
            return '\n'.join([f"{sedol}: {name} {weight:.3f}" for sedol, name, weight in zip(sedols, fund_names, w)])

        cursor = mplcursors.cursor(scatter, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            idx = sel.index
            sel.annotation.set_text(tooltip_text(idx))
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

        ax.set_xlabel('Risk (Standard Deviation)')
        ax.set_ylabel('Net Expected Return')
        ax.set_title('Efficient Frontier with Monte Carlo Simulation')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

        # Find tangent portfolio (maximum Sharpe ratio)
        risk_free_rate = 0.0525
        risks = [point['risk'] for point in frontier]
        yields = [point['net_yield'] for point in frontier]
        sharpe_ratios = [(y - risk_free_rate) / r for y, r in zip(yields, risks)]
        tangent_idx = np.argmax(sharpe_ratios)
        tangent_portfolio = frontier[tangent_idx]

        print("Tangent Portfolio (Max Sharpe Ratio):")
        for sedol, weight in zip(fd['sedol'], tangent_portfolio['weights']):
            print(f"  {sedol}: {weight:.4f}")
        print(f"  Net Yield: {tangent_portfolio['net_yield']:.4f}, Risk: {tangent_portfolio['risk']:.4f}, Sharpe: {sharpe_ratios[tangent_idx]:.4f}") 
        print("-" * 40)

        plt.tight_layout()

    def monte_carlo_efficient_frontier(self, n_portfolios=10000, risk_free_rate=0.0525, random_seed=42, ax=None, hoverover_legend=True):
        """
        Monte Carlo simulation to generate random portfolios and plot the efficient frontier.
        If ax is provided, plot on that axis.
        Returns: portfolio_risks, portfolio_returns, sharpe_ratios
        """
        np.random.seed(random_seed)
        n_assets = self.n_funds
        mean_returns = (self.funds_data['distribution_yield'] - self.funds_data['total_expenses']).values
        cov_matrix = self.covariance_matrix.values

        portfolio_returns = []
        portfolio_risks = []
        portfolio_weights = []

        for _ in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(n_assets))
            weights /= np.sum(weights)

            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            portfolio_returns.append(port_return)
            portfolio_risks.append(port_vol)
            portfolio_weights.append(weights)

        portfolio_returns = np.array(portfolio_returns)
        portfolio_risks = np.array(portfolio_risks)
        sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_risks

        if ax is not None:
            scatter = ax.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratios, cmap='viridis', marker='o', alpha=0.3, label='Monte Carlo Portfolios')
            plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')

            if hoverover_legend:
                # Add interactive hover tooltips for Monte Carlo portfolios
                sedols = list(self.funds_data['sedol'])
                def tooltip_text(index):
                    w = portfolio_weights[index]
                    return '\n'.join([f"{sedol}: {weight:.3f}" for sedol, weight in zip(sedols, w)])

                cursor = mplcursors.cursor(scatter, hover=True)
                @cursor.connect("add")
                def on_add(sel):
                    idx = sel.index
                    sel.annotation.set_text(tooltip_text(idx))
                    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

        return portfolio_risks, portfolio_returns, sharpe_ratios

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
print(f"Optimal Weights:\n{optimal_weights}")
print(f"Portfolio metrics: {metrics}")

# Calculate risk and net yield for the optimized portfolio
weights = optimal_weights['weight'].values
opt_net_yield = np.sum(weights * (fd['distribution_yield'] - fd['total_expenses']))
opt_risk = metrics['portfolio_risk']
print(f"Optimized Portfolio Net Yield: {opt_net_yield:.4f}, Risk: {opt_risk:.4f}")

# Generate efficient frontier
frontier = optimizer.generate_efficient_frontier()

# Print SEDOLs and weights for each portfolio on the efficient frontier
#print("Efficient Frontier Portfolios (SEDOLs and Weights):")
#for i, point in enumerate(frontier):
#    print(f"Portfolio {i+1}:")
#    for sedol, weight in zip(fd['sedol'], point['weights']):
#        print(f"  {sedol}: {weight:.4f}")
#    print(f"  Net Yield: {point['net_yield']:.4f}, Risk: {point['risk']:.4f}")
#    print("-" * 40)

# Create a single figure and axis
fig, ax = plt.subplots(figsize=(24, 18))

# Plot the optimized portfolio
ax.scatter([opt_risk], [opt_net_yield], color='blue', s=200, marker='*', label='Optimized Portfolio (Objective)', zorder=20)

#print (f"Efficient frontier: {frontier}")
# Plot Monte Carlo simulation on the same axis
optimizer.monte_carlo_efficient_frontier(ax=ax, hoverover_legend=False)

# Plot analytical efficient frontier on the same axis
optimizer.plot_efficient_frontier(frontier, ax=ax)

# Update legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()