import json
import requests
from bs4 import BeautifulSoup

def fetch_instrument_data(sedol):
    """
    Fetches and returns the instrument data for a given SEDOL from AJ Bell's website.
    """
    url = f"https://www.ajbell.co.uk/market-research/FUND:{sedol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
        if not script_tag:
            raise ValueError("__NEXT_DATA__ script tag not found")
        json_data = json.loads(script_tag.text)
        return json_data['props']['pageProps']['instrumentData']['instrument']
    except Exception as e:
        print(f"Error: {e}")
        return None
