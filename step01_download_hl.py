import sys
import requests
import csv
import time
import json
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError

@retry(wait=wait_exponential(multiplier=2, min=1, max=30), stop=stop_after_attempt(5))
def get_json_from_url(url):
    """
    Makes a GET request to a URL and returns the JSON response.
    Uses tenacity to retry on failures.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    # The JSONDecodeError that might be raised by response.json() will also trigger a retry
    return response.json()

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
        try:
            print("Fetching data structure...")
            data = get_json_from_url(base_url.format(0, chunk))
            if not data or "Results" not in data or not data["Results"]:
                print("\nCould not find any results in initial fetch. Aborting.")
                return
            
            column_names = list(data["Results"][0].keys())
            writer.writerow(column_names)
            print("Successfully fetched data structure. Starting download.")

        except RetryError as e:
            print(f"\nFailed to fetch initial data after multiple retries: {e}")
            return
        except (KeyError, IndexError) as e:
            print(f"\nCould not parse initial data structure from response: {e}")
            return

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

            try:
                data = get_json_from_url(url)
            except RetryError as e:
                print(f"\nFailed to retrieve data from {url} after multiple retries. Stopping download. Error: {e}")
                break

            # Extract the "Results" list from the JSON data
            results = data.get("Results")

            # Check if there are no more results
            if not results:
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

    print(f"\nData has been successfully saved to {csv_file}. Total funds downloaded: {total_funds}")


# Check if a CSV filename was provided as a command-line argument
if len(sys.argv) > 1:
    filename = sys.argv[1]
    scrape_funds_data(filename)
else:
    print("Please provide a CSV filename as a command-line argument.")