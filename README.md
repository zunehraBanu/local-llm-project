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

## 🔧 How to run it

### 1. Clone the repo and checkout the branch

```bash
git clone <your-repo-link>
cd <repo-folder>
git checkout -b webScraping

### 2. 📦 Install Dependencies
### Create and activate a virtual environment (optional but recommended), then run:
conda env create -f requirements.yml
conda activate business-news-web-scraper-env

### 3. 📝 Add Input URLs
### If you want to edit - Edit the input.txt file and include one URL per line. For example:
https://www.businesstoday.in/latest
https://www.financialexpress.com/market/

### 4. 🏃 Run the Script
python web_scraper.py
### After completion, check output.txt for the scraped headlines.
### All the best and happy scraping! 🕵️‍♀️📄