import sys
import csv
import requests
import time
import json
from bs4 import BeautifulSoup
import pandas as pd

properties_to_extract = [
    'alphas_oneYear',
    'alphas_threeYear',
    'alphas_fiveYear',
    'betas_oneYear',
    'betas_threeYear',
    'betas_fiveYear',
    'informationRatios_oneYear',
    'informationRatios_threeYear',
    'informationRatios_fiveYear',
    'rSquareds_oneYear',
    'rSquareds_threeYear',
    'rSquareds_fiveYear',
    'sharpeRatios_oneYear',
    'sharpeRatios_threeYear',
    'sharpeRatios_fiveYear',
    'standardDeviations_oneYear',
    'standardDeviations_threeYear',
    'standardDeviations_fiveYear',
    'trackingErrors_oneYear',
    'trackingErrors_threeYear',
    'trackingErrors_fiveYear'
]

fidelity_prefix = 'fidelity'

def extract_data(sedol):
    url = f"https://liveapi.yext.com/v2/accounts/me/answers/query?input={sedol}&experienceKey=fidelity-personal-investing-pi&api_key=44cc9a622358e7951c5ca3def99e2e0a&v=20220511&version=PRODUCTION&limit=%7B%22promotions%22%3A1%2C%22faqs%22%3A4%2C%22guidanceadvice_pi%22%3A4%2C%22investments%22%3A4%2C%22links%22%3A4%2C%22news%22%3A4%2C%22products%22%3A3%7D&locale=en_GB&sessionTrackingEnabled=false&referrerPageUrl=https%3A%2F%2Fwww.fidelity.co.uk%2F&source=STANDARD&jsLibVersion=v1.14.3"
    investment_data = {}

    try:
        response = requests.get(url)
        json_data = response.json()
    except Exception:
        json_data = ""

    if json_data and 'response' in json_data and 'modules' in json_data['response'] and isinstance(json_data['response']['modules'], list) and len(json_data['response']['modules']) > 0 and isinstance(json_data['response']['modules'][0], dict) and 'results' in json_data['response']['modules'][0] and isinstance(json_data['response']['modules'][0]['results'], list) and len(json_data['response']['modules'][0]['results']) > 0 and isinstance(json_data['response']['modules'][0]['results'][0], dict) and 'data' in json_data['response']['modules'][0]['results'][0]:
        investment_data = json_data['response']['modules'][0]['results'][0]['data']
    else:
        investment_data = None

    return investment_data

# Check if input_file and output_file were provided as command-line arguments
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:    
    print("Usage: python step02_fidelity_risk_ratings.py <input_file.csv>")
    sys.exit(1)

#with open(input_file, 'r') as file:
df = pd.read_csv(input_file)
#df.set_index('sedol', inplace=True)
#print(df.columns)

animation = "|/-\\"
success = 0
failure = 0
index = 0

# Process each sedol
total_funds = len(df['sedol'])
for sedol in df['sedol']:
    print(f"\rProcessing funds... {animation[index % len(animation)]} Fund {index}/{total_funds} Success/Failure {success}/{failure}", end='')
    sys.stdout.flush()

    # print(sedol)
    investment_data = extract_data(sedol)
    # print (investment_data)

    if investment_data:
        pi_url = investment_data.get('c_slugPI', None)
        c_isin = investment_data.get('c_isin', '')

    if pi_url:
        risk_rating_url = pi_url.replace('key-statistics', 'risk-and-rating')        
    else:        
        c_isin = ''
        risk_rating_url = ''

    # Process risk_rating_url if available
    if risk_rating_url:

        if risk_rating_url:
            df.loc[df['sedol'] == sedol, f"{fidelity_prefix}_risk_rating_url"] = risk_rating_url
        else:
            df.loc[df['sedol'] == sedol, f"{fidelity_prefix}_risk_rating_url"] = ''        

        response = requests.get(risk_rating_url)
    
        html_content = response.text

        if html_content:
            success += 1
        else:
            failure += 1

        # Extract JSON document from HTML response
        soup = BeautifulSoup(html_content, 'html.parser')
        script_tag = soup.find('script', id='__NEXT_DATA__', type='application/json')

        if script_tag:
            json_data = script_tag.string.strip()

            risk_measures = {}
            if json_data:
                try:
                    json_object = json.loads(json_data)
                    risk_measures = json_object['props']['pageProps']['initialState']['fund']['riskAndRating']['riskMeasures']
                except (json.JSONDecodeError, KeyError):
                    print(f"Failed to extract risk measures from URL: {risk_rating_url}")            

            # Extract and append the properties to the row
            for property_name in properties_to_extract:
                nested_properties = property_name.split('_')
                property_value = risk_measures
                for nested_property in nested_properties:
                    property_value = property_value.get(nested_property, '')
                    if not property_value:
                        break
                
                # print(f"{property_name} = {property_value}")
                # update dataframe
                if property_value:
                    df.loc[df['sedol'] == sedol, f"{fidelity_prefix}_{property_name}"] = float(property_value)                
        
        # save the file to disk
    df.to_csv(input_file, index=False, mode='w')
    index += 1
    