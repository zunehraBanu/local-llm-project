# gpt2_model.py

# Import tokenizer and model specific to GPT-2
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Import the base class
from base_model import BaseSentimentGenerator

# GPT2SentimentGenerator inherits all logic from BaseSentimentGenerator
class GPT2SentimentGenerator(BaseSentimentGenerator):
    def __init__(self):
        # Initialize the base class with GPT-2 model and tokenizer
        super().__init__("gpt2", GPT2Tokenizer, GPT2LMHeadModel)
