# gpt2_model.py

"""
This script loads the GPT-2 model, processes predefined prompts,
generates text responses, and then extracts the sentiment (Positive, Negative, or Neutral)
using a sentiment analysis pipeline. It saves the full details in 'gpt2_responses.txt'
and also writes just the sentiment labels into 'gpt2_sentiment_summary.txt'.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load the GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 small (117M parameters)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the prompts
prompts = [
    "A shocking Chinese AI advancement called DeepSeek is sending US stocks plunging",
    "As sales slump, Kohl’s turns to a new CEO to bring back customers",
    "Expect record-high egg prices for most of the year"
]

# Initialize the sentiment analysis pipeline using the cardiffnlp Twitter sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Mapping from model labels to human-readable sentiments
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# List to store just the sentiment labels
sentiment_list = []

# Open a file to save the responses along with their sentiments
with open("gpt2_responses.txt", "w", encoding="utf-8") as output_file:
    for prompt in prompts:
        # Encode the prompt and generate a response with a maximum length of 100 tokens
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs.input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
        # Decode the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Perform sentiment analysis on the generated response
        sentiment_result = sentiment_pipeline(response)[0]
        sentiment_label = label_map.get(sentiment_result["label"], sentiment_result["label"])
        sentiment_list.append(sentiment_label)
        
        # Write the full details to the response file
        output_file.write(f"Prompt: {prompt}\n")
        output_file.write(f"Response: {response}\n")
        output_file.write(f"Sentiment: {sentiment_label}\n\n")

# Write just the sentiment labels to a separate file (one per line)
with open("gpt2_sentiment_summary.txt", "w", encoding="utf-8") as summary_file:
    for sentiment in sentiment_list:
        summary_file.write(sentiment + "\n")

print("Full responses saved to 'gpt2_responses.txt' and sentiment summary saved to 'gpt2_sentiment_summary.txt'.")
