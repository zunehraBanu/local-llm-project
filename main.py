# main.py

# Import the custom classes from the GPT-2 and Phi files
from gpt2_model import GPT2SentimentGenerator
from phi_model import PhiSentimentGenerator

# Define a list of prompts to evaluate
prompts = [
    "A shocking Chinese AI advancement called DeepSeek is sending US stocks plunging",
    "As sales slump, Kohl’s turns to a new CEO to bring back customers",
    "Expect record-high egg prices for most of the year"
]

# Run GPT-2 based sentiment generation
print("Running GPT-2 Sentiment Analysis...")
gpt2_gen = GPT2SentimentGenerator()  # Create an instance of GPT2 class
gpt2_gen.process_prompts(prompts, "gpt2_responses.txt", "gpt2_sentiment_summary.txt")

# Run Phi-1.5 based sentiment generation
print("Running Phi-1.5 Sentiment Analysis...")
phi_gen = PhiSentimentGenerator()  # Create an instance of Phi class
phi_gen.process_prompts(prompts, "phi_responses.txt", "phi_sentiment_summary.txt")

