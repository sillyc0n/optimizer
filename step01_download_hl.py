import sys
import requests
import csv
import time

import json

def scrape_funds_data(csv_file):
    # URL for JSON data
        
    base_url = "https://www.hl.co.uk/ajax/funds/fund-search/search?investment=&companyid=&sectorid=&wealth=&unitTypePref=&tracker=&payment_frequency=&payment_type=&yield=&standard_ocf=&perf12m=&perf36m=&perf60m=&fund_size=&num_holdings=&start={}&rpp={}&lo=0&sort=fd.full_description&sort_dir=asc"
    
    # chunk size for the download
    chunk = 20

    # we will sleep for this number of seconds between API calls
    sleep = 0.4

    # TODO read existing csv file and update rather than override

    # Open the CSV file in write mode
    with open(csv_file, "w", newline="") as file:
        # Create a CSV writer
        writer = csv.writer(file)

        # Get the column names from the JSON response
        
        response = requests.get(base_url.format(0, chunk))
        if response.status_code == 200:
            data = response.json()
            column_names = list(data["Results"][0].keys())

            # Write the header row with column names
            writer.writerow(column_names)

            # Initialize the start parameter
            start = 0

            # Counter for downloaded funds
            total_funds = 0

            animation = "|/-\\"
            index = 0
            
            # Iterate until "ResultsReturned" is 0
            while True:
                
                progress = f"\rDownloading funds... {animation[index % len(animation)]} {total_funds} funds downloaded"
                sys.stdout.write(progress)
                sys.stdout.flush()

                # Construct the URL with the current start parameter
                url = base_url.format(start, chunk)

                # Send a GET request to the URL
                response = requests.get(url)

                # Check if the request was successful
                if response.status_code == 200:
                    # Get the JSON data from the response
                    #try:
                    data = response.json()
                    #except json.decoder.JSONDecodeError:
                    #    print(f"\nError decoding JSON response {response}")
                    #    continue                        

                    # Extract the "Results" list from the JSON data
                    results = data["Results"]

                    # Check if there are no more results
                    if len(results) == 0:
                        break

                    # Write each fund's data as a row in the CSV file
                    for fund in results:
                        writer.writerow(fund.values())
                        total_funds += 1

                    # Increment the start parameter by a chunk
                    start += chunk                        

                    # Update progress
                    index += 1

                    # Add a delay to not hammer the API
                    time.sleep(sleep)
                else:
                    print("\nFailed to retrieve data from the URL:", url)
                    break

    print("\nData has been successfully saved to", csv_file)


# Check if a CSV filename was provided as a command-line argument
if len(sys.argv) > 1:
    filename = sys.argv[1]
    scrape_funds_data(filename)
else:
    print("Please provide a CSV filename as a command-line argument.")

