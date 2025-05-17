import argparse
import os
import sys
import json
import time
import pandas as pd
import numpy as np
import random
from curl_cffi import requests
from tenacity import retry, wait_fixed, wait_exponential, stop_after_attempt

# constants
HOUR_IN_SECONDS = 3600
DAY_IN_SECONDS = HOUR_IN_SECONDS * 24

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
last_error  = None

# Set the headers for the API request
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) Gecko/20100101 Firefox/138.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Connection': 'keep-alive',
    'Cookie': 'GUC=AQABCAFoIExoR0IfIQSt&s=AQAAANmNwPa-&g=aB8CkQ; A1=d=AQABBH6-9GcCEFvabn4WJZcNtXhWvrsmuXwFEgABCAFMIGhHaPZ0rXYBAiAAAAcIe770Z6FcHpc&S=AQAAAndbvWR7_SyzOYJ529mHRIM; A3=d=AQABBH6-9GcCEFvabn4WJZcNtXhWvrsmuXwFEgABCAFMIGhHaPZ0rXYBAiAAAAcIe770Z6FcHpc&S=AQAAAndbvWR7_SyzOYJ529mHRIM; PRF=dock-collapsed%3Dtrue%26t%3D0P0001PP47.L%252B0P000019J5.L; _ebd=bid-9e7isk5jv9fjr&d=192819842f9a53fe97accdfade3d49b8&v=1; dflow=116; _dmit=BGWuRreWQbdOKUnpXQrgINUETYHakETYHaqCJsDvJAFAEQVABAAAAAAAAAAAAAAAAAAAAAUAAAlAAJtCA8gAEIAA.bid-9e7isk5jv9fjr.eyJvIjoiYmlkLTllN2lzazVqdjlmanIifQ==.1746862271253~AMEUCIQCJ8nt+8PVE/q/0qrNoYLn8VyrvATOWe+iJCulJJP6AywIgHMKdsjDQtHod2Mb2pXYIOuBCiMNP5Qqh+U8SeR6UCCg=; _dmieu=CQRMeYAQRMeYAAOABBENBpFgAAAAAAAAACiQAAAAAAAA.YAAAAAAAAAAA; A1S=d=AQABBH6-9GcCEFvabn4WJZcNtXhWvrsmuXwFEgABCAFMIGhHaPZ0rXYBAiAAAAcIe770Z6FcHpc&S=AQAAAndbvWR7_SyzOYJ529mHRIM; cmp=t=1747466470&j=1&u=1---&v=80; EuConsent=CQRMeYAQRMeYAAOADBENBpFgAAAAAAAAACiQAAAAAAAA',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Priority': 'u=0, i',
    'TE': 'trailers'
}

def positive_delay():
    while True:
        delay = np.random.normal(0.75, 0.5)
        if delay >= 0:
            return delay

total_funds = len(df)

if 'yahoo_timestamp' not in df.columns:
    df["yahoo_timestamp"] = None

if 'yahoo_symbol' not in df.columns:
    df["yahoo_symbol"] = None

#df['yahoo_timestamp'] = pd.to_datetime(df['yahoo_timestamp']).astype(int)

for sedol, symbol, yahoo_timestamp in zip(df['sedol'], df["yahoo_symbol"], df['yahoo_timestamp']):
    # Print progress information
    index += 1
    progress = f"\rProcessing symbols... {animation[index % len(animation)]} Processed: {index}/{total_funds} | Has Yahoo Symbol: {has_yahoo_symbol} | Has Yahoo Quotes: {has_yahoo_quotes} | Current symbol: {sedol}/{symbol} | Last Error: {last_error}"    
    print(progress, end='')
    sys.stdout.flush()

    # do not query yahoo quotes if what we have fetched within the last 24h    
    now = int(time.time())
    if yahoo_timestamp is not None and now - int(yahoo_timestamp) < DAY_IN_SECONDS:
        continue

    # get the symbol if not there
    if not symbol or pd.isna(symbol):
        delay = positive_delay()
        time.sleep(delay)
        # Make the API call to Yahoo Finance
        #url = f"https://query1.finance.yahoo.com/v1/finance/search?q={sedol}&lang=en-GB&region=GB&quotesCount=6&newsCount=4&enableFuzzyQuery=false&quotesQueryId=tss_match_phrase_query&multiQuoteQueryId=multi_quote_single_token_query&newsQueryId=news_cie_vespa&enableCb=true&enableNavLinks=true&enableEnhancedTrivialQuery=true&enableCulturalAssets=true&enableLogoUrl=true"
        #url = f"https://query2.finance.yahoo.com/v1/finance/search?q={sedol}&lang=en-US&region=US&quotesCount=6&newsCount=3&listsCount=2&enableFuzzyQuery=false&quotesQueryId=tss_match_phrase_query&multiQuoteQueryId=multi_quote_single_token_query&newsQueryId=news_cie_vespa&enableCb=false&enableNavLinks=true&enableEnhancedTrivialQuery=true&enableResearchReports=true&enableCulturalAssets=true&enableLogoUrl=true&enableLists=false&recommendCount=5&enablePrivateCompany=true"        
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={sedol}&lang=en-US&region=US&quotesCount=6&newsCount=3&listsCount=2&enableFuzzyQuery=false&quotesQueryId=tss_match_phrase_query&multiQuoteQueryId=multi_quote_single_token_query&newsQueryId=news_cie_vespa&enableCb=false&enableNavLinks=true&enableEnhancedTrivialQuery=true&enableResearchReports=true&enableCulturalAssets=true&enableLogoUrl=true&enableLists=false&recommendCount=5&enablePrivateCompany=true"
        response = requests.get(url, headers=headers, impersonate="chrome124")

        # Check if the response code is 200 (indicating a successful request)
        if response.status_code == 200:
            json_data = response.json()

            # Check if 'quotes' list is not empty and has at least one quote
            if json_data.get('quotes') and json_data['quotes'][0]:
                # Extract the symbol from the JSON response
                symbol = json_data['quotes'][0].get('symbol', None)
    
        if symbol != None and not pd.isna(symbol):
            # Add the symbol to the row
            df.loc[df['sedol'] == sedol, "yahoo_symbol"] = symbol
            # save the input csv file  
            df.to_csv(input_file, index=False, mode='w')
        else:
            last_error = f"No Yahoo symbol for sedol: {sedol}"

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
                response = requests.get(url, headers=headers, impersonate="chrome124")
                response.raise_for_status()
                return response
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                print(f"Error fetching data: {str(e)}")
                return None
        
        try:
            response = safe_get_request(url, headers)
        except requests.exceptions.RequestException as e:
            print(f'Failed after retrying: {e}')
        except NameError:
            pass
        
        if response is not None and response.status_code == 200:        
            # Save the response to a file
            output_filename = os.path.join(output_dir, f"{symbol}.json")
            with open(output_filename, 'w') as file:
                file.write(response.text)
            
            # update the timestamp
            df.loc[df['sedol'] == sedol, "yahoo_timestamp"] = int(time.time())            
            # save the input csv file
            df.to_csv(input_file, index=False, mode='w')

            # increase the counter
            has_yahoo_quotes += 1
        else:
            last_error = f"No quotes for yahoo symbol: {symbol}"
        
# Print completion message
completion_message = f"\nOutput saved to {input_file}"
print(completion_message)
