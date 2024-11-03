import pandas as pd
import pandasql as ps
import argparse

def process_data(hl_csv, correlation_csv, query_attr):
    hl_csv = pd.read_csv(hl_csv)
    hl_csv.set_index(['sedol', 'citicode'], inplace=True)

    corr_df = pd.read_csv(correlation_csv)

    # Create pandasql engine
    q = ps.sqldf

    # Execute SQL query
    return q(query_attr, locals())

# Usage
parser = argparse.ArgumentParser(description='Process HL CSV and Correlation Matrix')
parser.add_argument('hl_csv', help='HL CSV file path')
parser.add_argument('correlation_csv', help='Correlation matrix CSV file path')
parser.add_argument('query_attr', help='SQL query attribute')

args = parser.parse_args()
result = process_data(args.hl_csv, args.correlation_csv, args.query_attr)

print(result)
