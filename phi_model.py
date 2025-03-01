# phi_model.py

"""
This script loads the Microsoft Phi-1.5 model, processes predefined prompts,
generates text responses, and then extracts the sentiment (Positive, Negative, or Neutral)
using a sentiment analysis pipeline. It saves the full details in 'phi_responses.txt'
and also writes just the sentiment labels into 'sentiment_summary.txt'.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the Phi-1.5 model and tokenizer
model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if CUDA (GPU) is available and use it if possible
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
with open("phi_responses.txt", "w", encoding="utf-8") as output_file:
    for prompt in prompts:
        # Encode the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Generate a response with a maximum length of 100 tokens
        outputs = model.generate(inputs.input_ids, max_length=100)
        # Decode the response text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Perform sentiment analysis on the generated response
        sentiment_result = sentiment_pipeline(response)[0]
        # Map the label to a friendly string
        sentiment_label = label_map.get(sentiment_result["label"], sentiment_result["label"])
        sentiment_list.append(sentiment_label)
        
        # Write the prompt, response, and sentiment into the file
        output_file.write(f"Prompt: {prompt}\n")
        output_file.write(f"Response: {response}\n")
        output_file.write(f"Sentiment: {sentiment_label}\n\n")

# Write just the sentiment labels to a separate file, one per line.
with open("sentiment_summary.txt", "w", encoding="utf-8") as summary_file:
    for sentiment in sentiment_list:
        summary_file.write(sentiment + "\n")

print("Full responses saved to 'phi_responses.txt' and sentiment summary saved to 'sentiment_summary.txt'.")
