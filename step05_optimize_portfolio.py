import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Calculate Efficient Frontier')
parser.add_argument('hl_csv', help='HL funds CSV file path')
parser.add_argument('cm_csv', help='Correlation matrix CSV file path')
parser.add_argument('sedols', nargs='+', type=str, help='List of SEDOLs in portfolio')
#parser.add_argument('output_file', default='efficient_frontier.png', help='Output plot file path')
args = parser.parse_args()

# Load data and create covariance matrix
hl_csv = pd.read_csv(args.hl_csv)
cov_matrix = pd.read_csv(args.cm_csv)

def optimize_weights(weights):
    # Implement optimization logic here
    pass

print (args.sedols)

# Run optimization
#optimal_weights = minimize(optimize_weights, x0=np.array([1./len(portfolio)]*len(portfolio)))

# Plot efficient frontier
plt.plot()
plt.show()

#plt.savefig(args.output_file)
