# project3_main.py

from gpt2_model import GPT2SentimentGenerator
# OR
# from phi_model import PhiSentimentGenerator

def main():
    # Step 1: Read the headlines from project 2 output file
    with open('input.txt', 'r', encoding='utf-8') as f:
        headlines = [line.strip() for line in f if line.strip()]

    # Step 2: Initialize the Sentiment Generator (we can choose GPT-2 or Phi)
    sentiment_generator = GPT2SentimentGenerator()
    # sentiment_generator = PhiSentimentGenerator()  # we can switch if needed

    # Step 3: Process prompts and save only sentiments
    sentiments = []
    for headline in headlines:
        _, sentiment = sentiment_generator.analyze_prompt(headline)
        sentiments.append(sentiment)

    # Step 4: Write sentiments to output file
    with open('project3_sentiments.txt', 'w', encoding='utf-8') as f:
        for sentiment in sentiments:
            f.write(sentiment + '\n')

    print("Project 3 sentiment analysis completed. Check project3_sentiments.txt")

if __name__ == "__main__":
    main()
