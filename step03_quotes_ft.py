import argparse
import pandas as pd
import requests
import time
from urllib.parse import quote
import sys
from bs4 import BeautifulSoup
import requests
import AJBellFund

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}
panels = {
    "modriskmeasures1y-panel",
    "modriskmeasures3y-panel",
    "modriskmeasures5y-panel"
}

def search_ft(symbol):
    url = f"https://markets.ft.com/data/searchapi/searchsecurities?query={quote(symbol)}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_risk_data(symbol):
    url = f'https://markets.ft.com/data/funds/tearsheet/risk?s={quote(symbol)}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    measures = {}
    
    for panel in ['1y', '3y', '5y']:
        table_container = soup.find('div', id=f"modriskmeasures{panel}-panel")
        if table_container:
            tables = table_container.find_all('table')
            
            for i, table in enumerate(tables):
                trs = table.find('tbody').find_all('tr')
                for tr in trs:
                    prop_name = tr.find_all('td')[0].text.strip().lower().replace(' ', '_')
                    prop_value = tr.find_all('td')[1].text.strip()           
                
                    measures[f'ft_{prop_name}_{panel}'] = prop_value
    
    return {**measures}

def process_csv(filename):
    df = pd.read_csv(filename)
    df['ft_symbol'] = pd.NA
    df['ft_xid'] = pd.NA

    total = len(df)
    errors = 0
    for index, row in df.iterrows():
        # Progress output
        print(f"\rProcessing row {index+1} of {total}. Errors: {errors}", end='', flush=True)
        
        # TODO - check ajbell for isin

        # Fill c_isin if empty
        #if pd.isna(row['c_isin']) or not str(row['c_isin']).strip():
        #    c_isin = f"GB00{row['sedol']}"
        #else:
        #    c_isin = row['c_isin']

        if pd.isna(row['isin']) or not str(row['isin']).strip():
            sedol = row['sedol']
            json = AJBellFund.fetch_instrument_data(sedol)
            if json is None:
                isin = None
            else:
                isin = json['isin']
        else:
            isin = row['isin']

        if isin is not None:
            ft_data = search_ft(isin)

        if ft_data and ft_data.get('data') and ft_data['data'].get('security'):
            security = ft_data['data']['security'][0]
            df.at[index, 'ft_symbol'] = security.get('symbol')
            df.at[index, 'ft_xid'] = security.get('xid')
            df.at[index, 'isin'] = isin
        else:
            errors += 1
            continue

        for key, value in get_risk_data(df.at[index, 'ft_symbol']).items():
            df.at[index, key] = pd.to_numeric(value, errors='coerce')
        
        # Save the updated DataFrame back to CSV
        df.to_csv(filename, index=False)

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.2)

    print(f"Processed {total} rows.")    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process HL CSV with FT API lookup')
    parser.add_argument('hl_csv', type=str, help='CSV file name to process')
    args = parser.parse_args()
    process_csv(args.hl_csv)
