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
