import argparse
import os
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Calculate Correlation Matrix for CSV Files')
parser.add_argument('directory', help='Directory containing CSV files with quotes from Yahoo')
parser.add_argument('correlation_csv', help='Output filename for correlation matrix CSV file')
args = parser.parse_args()

# Get the list of CSV files in the specified directory
json_files = [file for file in os.listdir(args.directory) if file.endswith('.json')]

# Initialize a dictionary to store 'Adj Close' values from each file
adj_close_values = {}

# Progress animation variables
animation = "|/-\\"
index = 0

temp_dfs = []

# TODO - this code should only consider funds in hl.csv that have quotes in json 
# the risk is that it may calculate matrix for old funds not on hl anymore

# Process each file and extract 'Adj Close' values
for filename in json_files:
    file_path = os.path.join(args.directory, filename)    

    # Read the CSV file and extract Adjusted Close
    json = pd.read_json(file_path, dtype_backend='numpy_nullable')

    # Select the specific nested data for index and columns
    timestamp = json['chart']['result'][0]['timestamp']
    column_data = json['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']
    symbol = json['chart']['result'][0]['meta']['symbol']

    # Add data to combined DataFrame
    temp_df = pd.DataFrame({symbol: column_data}, index=timestamp)
    temp_dfs.append(temp_df)

    # processing finished
    index += 1

    # Print progress information
    progress = f"\rProcessing symbols... {animation[index % len(animation)]} Processed: {index}/{len(json_files)} Current symbol: {symbol}"
    print(progress, end='')    

print("Calculating Matrix. Please wait ...")

# Combine all temporary DataFrames outside the loop
combined_df = pd.concat(temp_dfs, axis=1, join='outer')

# Output the data frame to a file
#output_file = os.path.join(args.directory, "df.csv")
#combined_df.to_csv(output_file, index=True)
#print(f"combined_df saved to {output_file}")

# Create a DataFrame from the 'Adj Close' values
df_adj_close = pd.DataFrame(combined_df)

# Calculate the correlation matrix
correlation_matrix = df_adj_close.corr()

# Output the correlation matrix to a file
output_file = os.path.join(args.correlation_csv)
correlation_matrix.to_csv(output_file, index=True)

print(f"Correlation matrix saved to {output_file}")

