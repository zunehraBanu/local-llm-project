# web_scraper.py

"""
This script reads business news URLs from 'input.txt',
scrapes headlines from each, and writes them to 'output.txt'.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# --- Initialize the Selenium WebDriver in headless mode (no browser window will pop up) ---
def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    driver = webdriver.Chrome(options=chrome_options)
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
