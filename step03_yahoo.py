import argparse
import os
import sys
#import csv
import requests
import json
import time
import pandas as pd

# Check if input_file and output_dir were provided as command-line arguments
if len(sys.argv) > 2:
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
else:    
    print("Usage: python step03_yahoo.py <input_file.csv> <output_dir")
    sys.exit(1)

df = pd.read_csv(input_file, dtype={'yahoo_symbol': str})

# Progress animation variables
animation = "|/-\\"
index = 0

# Open the output file in append mode
total_funds = len(df)
print(df)
for sedol in df['sedol']:
    
    #symbol = df.loc[df['sedol'] == sedol, "yahoo_symbol"]
            
    # Set the headers for the API request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
    }

    # Make the API call to Yahoo Finance
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={sedol}&lang=en-GB&region=GB&quotesCount=6&newsCount=4&enableFuzzyQuery=false&quotesQueryId=tss_match_phrase_query&multiQuoteQueryId=multi_quote_single_token_query&newsQueryId=news_cie_vespa&enableCb=true&enableNavLinks=true&enableEnhancedTrivialQuery=true&enableCulturalAssets=true&enableLogoUrl=true"

    response = requests.get(url, headers=headers)

    # Check if the response code is 200 (indicating a successful request)
    if response.status_code == 200:
        json_data = response.json()

        # Check if 'quotes' list is not empty and has at least one quote
        if json_data.get('quotes') and json_data['quotes'][0]:
            # Extract the symbol from the JSON response
            symbol = json_data['quotes'][0].get('symbol', 'N/A')

    # Add the symbol to the row
    df.loc[df['sedol'] == sedol, "yahoo_symbol"] = symbol
    # save csv    
    df.to_csv(input_file, index=False, mode='w')

    # download quotes
    period2 = int(time.time())
    period1 = period2 - (365 * 24 * 60 * 60)

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={period1}&period2={period2}&interval=1d&includePrePost=true&events=div%7Csplit%7Cearn&&lang=en-GB&region=GB"
    response = requests.get(url, headers=headers)
    
    output_filename = os.path.join(output_dir, f"{symbol}.json")
    
    if response.status_code == 200:
        # Save the response to a file
        with open(output_filename, 'w') as file:
            file.write(response.text)

    index += 1

    time.sleep(0.2)
        
    # Print progress information
    progress = f"\rProcessing symbols... {animation[index % len(animation)]} Processed: {index}/{total_funds} Current symbol: {sedol}/{symbol}"
    print(progress, end='')
    sys.stdout.flush()

# Print completion message
completion_message = f"\nOutput saved to {input_file}"
print(completion_message)