# phi_model.py

# Import tokenizer and model for Microsoft Phi-1.5
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import the base class
from base_model import BaseSentimentGenerator

# PhiSentimentGenerator inherits from the base class
class PhiSentimentGenerator(BaseSentimentGenerator):
    def __init__(self):
        # Initialize the base class with Phi-1.5 model and tokenizer
        super().__init__("microsoft/phi-1_5", AutoTokenizer, AutoModelForCausalLM)

