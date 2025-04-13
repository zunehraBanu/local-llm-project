# web_scraper.py

"""
This script reads business news URLs from 'input.txt',
scrapes headlines from each, and writes them to 'output.txt'.
"""
# Selenium is used to automate browser actions like loading pages and retrieving HTML
from selenium import webdriver
# Options lets us configure how Chrome behaves (e.g., headless mode)
from selenium.webdriver.chrome.options import Options
# BeautifulSoup is a library for parsing HTML and extracting data
from bs4 import BeautifulSoup
# Time is used to pause the script briefly so that dynamic web content can load
import time

# --- Initialize the Selenium WebDriver in headless mode (no browser window will pop up) ---
def init_driver():
    chrome_options = Options() # Create a ChromeOptions object to set browser preferences
    chrome_options.add_argument("--headless") # Run in headless mode (no browser UI)
    chrome_options.add_argument("--disable-gpu") # Disable GPU usage for compatibility
    chrome_options.add_argument("--ignore-certificate-errors") # Ignore SSL certificate errors
    chrome_options.add_argument("--allow-running-insecure-content") # Allow HTTP content on HTTPS pages
    chrome_options.add_argument("--log-level=3")  # Suppress logs
    driver = webdriver.Chrome(options=chrome_options) # Launch Chrome with these settings
    return driver

# --- Read URLs from the input file ---
def read_urls(file_path='input.txt'):
    with open(file_path, 'r') as f:
        # Remove empty lines and strip whitespace
        return [line.strip() for line in f if line.strip()]

# --- Scrape headlines from a single URL ---
def scrape_headlines(driver, url):
    headlines = []
    try:
        driver.get(url)  # Load the web page
        time.sleep(2)  # Wait for the page to fully load
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Look through various heading tags (h1 to h6)
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for item in soup.find_all(tag):
                text = item.get_text().strip()
                # Filter out very short headings (less than 1 characters)
                if len(text) > 1:
                    headlines.append(text)
    except Exception as e:
        print(f"Error scraping {url}: {e}")  # Show error if something goes wrong
    return headlines

# --- Write all collected headlines to the output file ---
def write_output(headlines, output_file='output.txt'):
    with open(output_file, 'w', encoding='utf-8') as f:
        for headline in headlines:
            f.write(headline + '\n')

# --- Main script logic ---
def main():
    driver = init_driver()  # Start browser
    urls = read_urls()      # Load URLs from file
    all_headlines = []      # Store all headlines in this list

    # Loop through each URL and scrape headlines
    for url in urls:
        print(f"Scraping: {url}")
        headlines = scrape_headlines(driver, url)
        print(f"→ Found {len(headlines)} headlines\n")  # Print number of headlines found
        all_headlines.extend(headlines)  # Add to the full list

    driver.quit()  # Close browser
    write_output(all_headlines)  # Save results
    print("✅ Scraping completed! Check output.txt.")

# --- Run the main function when the script is executed ---
if __name__ == '__main__':
    main()
