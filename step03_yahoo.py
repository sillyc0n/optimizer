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
has_yahoo_quotes = 0
has_yahoo_symbol = 0

# Set the headers for the API request
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
}

total_funds = len(df)
for sedol in df['sedol']:
         
    symbol = df.loc[df['sedol'] == sedol, "yahoo_symbol"].iloc[0]

    # get the symbol if not there
    if not symbol:
        time.sleep(0.5)
        # Make the API call to Yahoo Finance
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={sedol}&lang=en-GB&region=GB&quotesCount=6&newsCount=4&enableFuzzyQuery=false&quotesQueryId=tss_match_phrase_query&multiQuoteQueryId=multi_quote_single_token_query&newsQueryId=news_cie_vespa&enableCb=true&enableNavLinks=true&enableEnhancedTrivialQuery=true&enableCulturalAssets=true&enableLogoUrl=true"
        response = requests.get(url, headers=headers)

        # Check if the response code is 200 (indicating a successful request)
        if response.status_code == 200:
            json_data = response.json()

            # Check if 'quotes' list is not empty and has at least one quote
            if json_data.get('quotes') and json_data['quotes'][0]:
                # Extract the symbol from the JSON response
                symbol = json_data['quotes'][0].get('symbol', None)
    
        if symbol:
            # Add the symbol to the row or reset it to blank
            df.loc[df['sedol'] == sedol, "yahoo_symbol"] = symbol
            # save csv    
            df.to_csv(input_file, index=False, mode='w')

    # get the quotes
    if symbol:
        has_yahoo_symbol += 1
        # download quotes
        period2 = int(time.time())
        period1 = period2 - (365 * 24 * 60 * 60)

        time.sleep(0.5)
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={period1}&period2={period2}&interval=1d&includePrePost=true&events=div%7Csplit%7Cearn&&lang=en-GB&region=GB"
        response = requests.get(url, headers=headers)
    
        if response.status_code == 200:
            has_yahoo_quotes += 1
            # Save the response to a file
            output_filename = os.path.join(output_dir, f"{symbol}.json")
            with open(output_filename, 'w') as file:
                file.write(response.text)
        else:
            print(f"No Yahoo Quotes for yahoo symbol: {symbol} url: {url}")
    else:
        print(f"No Yahoo Symbol for sedol: {sedol} url {url}")

    index += 1    
        
    # Print progress information
    progress = f"\rProcessing symbols... {animation[index % len(animation)]} Processed: {index}/{total_funds} | Has Yahoo Symbol: {has_yahoo_symbol} | Has Yahoo Quotes: {has_yahoo_quotes} | Current symbol: {sedol}/{symbol}"
    print(progress, end='')
    sys.stdout.flush()

# Print completion message
completion_message = f"\nOutput saved to {input_file}"
print(completion_message)