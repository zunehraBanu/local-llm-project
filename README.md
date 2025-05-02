# Project 3: Web Scraping and Sentiment Analysis

# Overview
This project combines **web scraping** and **local language model inference** to perform sentiment analysis on news headlines.

The project:
- Reads scraped headlines from a file (output.txt generated in Project 2).
- Uses a fine-tuned **GPT-2** model for sentiment analysis.
- Outputs the predicted sentiments (Positive, Neutral, Negative) into a text file (project3_sentiments.txt).

Additionally, two test cases are written using **pytest** to ensure:
- The input file exists.
- The input file contains data.

# Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.8** (or compatible version)
- **Conda** (Miniconda or Anaconda)
- **PyTorch** and **transformers** libraries
- **pytest** for running the test cases
- A stable internet connection for the first run (to download model weights)

Install the required libraries using:

pip install torch transformers pytest

# Setup Instructions

## 1. Clone the Repository
Open your terminal or command prompt and run:  
git clone https://github.com/yourusername/your-repo-name.git

Then, navigate to the project directory:  
cd your-repo-name

## 2. Create and Activate the Conda Environment
Use the provided requirements.yaml file to set up the environment. Run:  
conda env create -f requirements.yaml

Activate the environment:  
conda activate cs_project1

## 3. (Optional) Install Additional Dependencies
All necessary packages are specified in the requirements.yaml. If you need to install any additional Python packages, you can use:  
pip install package_name

# Project Structure
project3/  
├── project3_main.py         # Main script to perform sentiment analysis  
├── base_model.py            # Base model class (for consistency)  
├── gpt2_model.py            # Fine-tuned GPT-2 sentiment analysis model  
├── phi_model.py             # (Optional) Phi-1.5 model (not used in this project)  
├── web_scraper.py           # (Optional) Web scraping script (Project 2)  
├── input.txt               # Input headlines file (from Project 2)  
├── project3_sentiments.txt  # Output sentiments  
├── test_project3.py         # Pytest test cases 
├── p1input.txt             # project 1 input
└── README.md                # This file  

# How to Run
1. Ensure you have the output.txt file in the same directory. This file should contain the scraped headlines from Project 2.

2. Run the main script:

python project3_main.py

You should see the following message:

✅ Project 3 sentiment analysis completed. Check project3_sentiments.txt

The predicted sentiments will be saved in project3_sentiments.txt.

# Test Cases
Two test cases are included in test_sentiment_analysis.py:

1. Test if input.txt exists  
2. Test if input.txt contains data (is not empty)

To run the tests, use:

pytest test_sentiment_analysis.py

Expected(not exact) output if both tests pass:

============================= test session starts =============================  
collected 2 items

test_sentiment_analysis.py ..                                                           [100%]

============================== 2 passed in Xs ================================

# All the best!