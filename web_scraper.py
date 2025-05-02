# Import necessary modules
from selenium import webdriver  # To interact with the browser via Selenium WebDriver
from selenium.webdriver.chrome.options import Options  # For setting browser options like headless mode
from selenium.webdriver.common.by import By  # For locating elements on the webpage
from selenium.webdriver.support.ui import WebDriverWait  # For waiting for elements to load
from selenium.webdriver.support import expected_conditions as EC  # For checking element's presence or condition
from bs4 import BeautifulSoup  # For parsing HTML content easily
import time  # To add delays for waiting or simulating human behavior
import re  # For regular expression operations to clean up the text

# Base class for Scraper (acts as an interface in this case)
class ScraperBase:
    def __init__(self, url_list):
        """Initializes with a list of URLs to scrape."""
        self.urls = url_list  # Store the list of URLs to scrape

    def scrape(self):
        """This method is meant to be implemented by subclasses to perform scraping."""
        raise NotImplementedError("Subclasses should implement this method.")  # Ensure that the method is implemented by subclasses

    def write_output(self, headlines, output_file='output.txt'):
        """Writes the extracted headlines to an output file."""
        with open(output_file, 'w', encoding='utf-8') as f:  # Open the output file in write mode with UTF-8 encoding
            for headline in headlines:  # Loop through each headline in the list
                f.write(headline + '\n')  # Write the headline to the file with a newline after each

# WebScraper class inherits from ScraperBase
class WebScraper(ScraperBase):
    def __init__(self, url_list):
        """Initializes the WebScraper with the list of URLs to scrape and sets up the web driver."""
        super().__init__(url_list)  # Call the constructor of ScraperBase
        self.driver = self.init_driver()

    def init_driver(self):
        """Sets up the Chrome WebDriver in headless mode."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run without opening the browser UI
        chrome_options.add_argument("--disable-gpu")  # Disable GPU for compatibility
        chrome_options.add_argument("--ignore-certificate-errors")  # Ignore certificate errors
        chrome_options.add_argument("--allow-running-insecure-content")  # Allow insecure content
        chrome_options.add_argument("--log-level=3")  # Suppress logs
        driver = webdriver.Chrome(options=chrome_options)  # Start the browser with these options
        return driver

    def scrape(self):
        """Main method to scrape headlines from the list of URLs."""
        all_headlines = []  # List to store all extracted headlines
        for url in self.urls:
            print(f"Scraping: {url}")
            headlines = self.scrape_headlines(url)  # Extract headlines from the current URL
            print(f"→ Found {len(headlines)} headlines\n")
            all_headlines.extend(headlines)  # Add the extracted headlines to the master list
        self.write_output(all_headlines)  # Write all headlines to the output file
        self.driver.quit()  # Close the browser after scraping
        print("✅ Scraping completed! Check output.txt.")

    def scrape_headlines(self, url):
        """Scrapes headlines from a single webpage, filtering unwanted headlines."""
        headlines = []  # List to store headlines from the current page
        try:
            self.driver.get(url)  # Navigate to the webpage
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )  # Wait for the body tag to be present (indicating page load)
            
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # Scroll down the page
            time.sleep(2)  # Wait for additional content to load (if any)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')  # Parse the page with BeautifulSoup
            # Iterate over all possible header tags (h1, h2, h3, h4, h5, h6)
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                for item in soup.find_all(tag):  # Find each headline tag
                    text = item.get_text().strip()  # Extract and clean the headline text

                    # Clean the text by removing non-alphanumeric characters and multiple spaces
                    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

                    # Check if the cleaned headline meets the following criteria:
                    # 1. It is not too short (1-2 words)
                    # 2. It does not end with an ellipsis ("...")
                    if len(cleaned_text.split()) >= 3 and not cleaned_text.endswith("..."):
                        headlines.append(cleaned_text)  # Add the valid headline to the list
        except Exception as e:
            print(f"Error scraping {url}: {e}")  # Handle any errors during scraping
        return headlines  # Return the list of headlines for the current page

# --- Main script logic ---
def main():
    """Main function that loads URLs from a file, creates a scraper instance, and starts scraping."""
    with open('input.txt', 'r') as f:
        # Read the URLs from the input file and strip any extra whitespace
        urls = [line.strip() for line in f if line.strip()]
    
    # Create an instance of WebScraper with the list of URLs
    scraper = WebScraper(urls)
    # Start the scraping process
    scraper.scrape()

# --- Run the main function when the script is executed ---
if __name__ == '__main__':
    main()
