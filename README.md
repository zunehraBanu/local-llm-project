# Business News Web Scraper

This Python program scrapes **headlines from business news websites** using Selenium and BeautifulSoup.

---

## 🧠 What it does

- Reads business news URLs from `input.txt`
- Uses Selenium (headless browser) to load each site
- Scrapes headings from tags like `<h1>`, `<h2>`, and `<h3>`
- Cleans and saves the headlines to `output.txt`

---

## 📂 Files in this branch

- `web_scraper.py` - main script
- `input.txt` - contains the URLs to scrape
- `output.txt` - contains scraped headlines (auto-generated)
- `requirements.yml` - list of required packages

---
## Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.8** (or compatible version)
- **Conda** (Miniconda or Anaconda)
- A stable internet connection for the first run (to scrape websites)

## Setup Instructions

### 1. Clone the Repository
Open your terminal or command prompt and run:
<code>git clone https://github.com/yourusername/your-repo-name.git</code>

Then, navigate to the project directory:
<code>cd your-repo-name</code>

### 2. Create and Activate the Conda Environment
Use the provided `requirements.yaml` file to set up the environment. Run:
<code>conda env create -f requirements.yml</code>

Activate the environment:
<code>conda activate business-news-web-scraper-env</code>

### 3. (Optional) Install Additional Dependencies
All necessary packages are specified in the `requirements.yaml`. If you need to install any additional Python packages, you can use:
<code>pip install package_name</code>

## Running the Scraper


### 1. 📝 Add Input URLs
### If you want to edit - Edit the input.txt file and include one URL per line. For example:
https://www.businesstoday.in/latest

https://www.financialexpress.com/market/

### 2. 🏃 Run the Script
<code>python web_scraper.py</code>
### After completion, check output.txt for the scraped headlines.
## All the best and happy scraping! 🕵️‍♀️📄