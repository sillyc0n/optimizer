import argparse
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Find securities with correlation between 0 and 0.4 for selected securities')
parser.add_argument('securities_csv', help='CSV file containing the list of securities')
parser.add_argument('correlation_csv', help='CSV file containing the correlation matrix')
parser.add_argument('attribute', help='Attribute to sort by (e.g., perf3m, perf6m, sharpeRatios.oneYear)')
parser.add_argument('top_n', type=int, help='Number of top entries to select')
args = parser.parse_args()

# Read the securities CSV file
securities_df = pd.read_csv(args.securities_csv)

# Filter out securities with net_annual_charge < 0.5
selected_securities = securities_df[securities_df['net_annual_charge'] < 0.5]

# Sort the selected securities by the specified attribute in descending order
sorted_securities = selected_securities.sort_values(by=args.attribute, ascending=False)

# Take the top n entries
top_n_securities = sorted_securities[sorted_securities['yahoo_symbol'].notnull()].head(args.top_n)

# Read correlation matrix CSV file
correlation_matrix = pd.read_csv(args.correlation_csv, index_col=0)

# Print selected security details
print("Selected Securities:")
for index, security in top_n_securities.iterrows():
    sedol = security['sedol']
    citicode = security['citicode']
    full_description = security['full_description']
    selected_attribute = security[args.attribute]
    annual_charge = security['annual_charge']

    print(f"Sedol: {sedol}, Citicode: {citicode}, Full Description: {full_description}, annual_charge: {annual_charge} {args.attribute}: {selected_attribute}")

# Find securities with correlation between 0 and 0.4 for each security in top_n_securities
for index, security in top_n_securities.iterrows():
    yahoo_symbol = security['yahoo_symbol']

    correlated_securities = correlation_matrix.loc[yahoo_symbol][(correlation_matrix.loc[yahoo_symbol] <= 0.2)]
    correlated_securities = correlated_securities.index.tolist()
    correlated_securities = [sec for sec in correlated_securities if sec in selected_securities['yahoo_symbol'].values]
    correlated_securities = selected_securities[selected_securities['yahoo_symbol'].isin(correlated_securities)]
    correlated_securities = correlated_securities.sort_values(by=args.attribute, ascending=False).head(8)

    print(f"Least correlated securities to {yahoo_symbol}, {security['sedol']}")
    for index, correlated_security in correlated_securities.iterrows():
        sedol = correlated_security['sedol']
        citicode = correlated_security['citicode']
        full_description = correlated_security['full_description']
        correlation_value = correlation_matrix.loc[yahoo_symbol, correlated_security['yahoo_symbol']]

        print(f"Sedol: {sedol}, Citicode: {citicode}, Full Description: {full_description}, Correlation: {correlation_value}, Selected Attribute: {args.attribute}: {correlated_security[args.attribute]}")
