import argparse
import os
import sys
import requests
import json
import time
import pandas as pd
import numpy as np
import random
import requests
from tenacity import retry, wait_fixed, wait_exponential, stop_after_attempt

# Check if input_file and output_dir were provided as command-line arguments
if len(sys.argv) > 2:
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
else:    
    print("Usage: python step03_yahoo.py <input_file.csv> <output_dir")
    sys.exit(1)

df = pd.read_csv(input_file, dtype={'yahoo_symbol': str})
df = df.replace({np.nan: None})

# Progress animation variables
animation = "|/-\\"
index = 0
has_yahoo_quotes = 0
has_yahoo_symbol = 0

# Set the headers for the API request
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
}

def positive_delay():
    while True:
        delay = np.random.normal(0.75, 0.5)
        if delay >= 0:
            return delay

total_funds = len(df)
for sedol in df['sedol']:    

    if 'yahoo_symbol' in df.columns:
        symbol = df.loc[df['sedol'] == sedol, "yahoo_symbol"].iloc[0]
    else:
        symbol = None

    # get the symbol if not there
    if not symbol or pd.isna(symbol):
        delay = positive_delay()
        time.sleep(delay)
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
    
        if symbol != None and not pd.isna(symbol):
            # Add the symbol to the row or reset it to blank
            df.loc[df['sedol'] == sedol, "yahoo_symbol"] = symbol
            # save csv    
            df.to_csv(input_file, index=False, mode='w')
        else:
            print(f"No Yahoo Symbol for sedol: {sedol} url {url}")

    # get the quotes
    if symbol != None and not pd.isna(symbol):
        has_yahoo_symbol += 1
        # download quotes
        period2 = int(time.time())
        period1 = period2 - (365 * 24 * 60 * 60)

        time.sleep(0.5)
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={period1}&period2={period2}&interval=1d&includePrePost=true&events=div%7Csplit%7Cearn&&lang=en-GB&region=GB"
        
        @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2.5, min=4, max=10000))
        def safe_get_request(url, headers):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError):
                    print(f"HTTP Error: {e}")
                else:
                    print(f"Other error: {e}")
                raise
        
        try:
            response = safe_get_request(url, headers)
        except requests.exceptions.RequestException as e:
            print(f'Failed after retrying: {e}')
        
        if response.status_code == 200:
            df.loc[df['sedol'] == sedol, "yahoo_quotes"] = True
            has_yahoo_quotes += 1
            # Save the response to a file
            output_filename = os.path.join(output_dir, f"{symbol}.json")
            with open(output_filename, 'w') as file:
                file.write(response.text)
        else:
            df.loc[df['sedol'] == sedol, "yahoo_quotes"] = False
            print(f"No Yahoo Quotes for yahoo symbol: {symbol} url: {url}")
    else:
        df.loc[df['sedol'] == sedol, "yahoo_quotes"] = False
        print(f"No Yahoo Quotes for yahoo symbol: {symbol} url: {url}")

    index += 1
        
    # Print progress information
    progress = f"\rProcessing symbols... {animation[index % len(animation)]} Processed: {index}/{total_funds} | Has Yahoo Symbol: {has_yahoo_symbol} | Has Yahoo Quotes: {has_yahoo_quotes} | Current symbol: {sedol}/{symbol}"
    print(progress, end='')
    sys.stdout.flush()

# Print completion message
completion_message = f"\nOutput saved to {input_file}"
print(completion_message)
