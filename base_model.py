# base_model.py

import torch
from transformers import pipeline

# Base class that contains shared logic for any sentiment generation model
class BaseSentimentGenerator:
    def __init__(self, model_name, tokenizer_cls, model_cls):
        # Use GPU if available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the tokenizer for the specified model
        self.tokenizer = tokenizer_cls.from_pretrained(model_name)
        
        # Load the model and move it to the selected device
        self.model = model_cls.from_pretrained(model_name).to(self.device)
        
        # Load a pre-trained sentiment analysis pipeline
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        
        # Map model-specific sentiment labels to human-friendly labels
        self.label_map = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }

    # Generate a response for a given prompt and get its sentiment
    def analyze_prompt(self, prompt, max_length=100):
        # Tokenize the input prompt and move it to the device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask if "attention_mask" in inputs else None

        # 🛠 Fix: Handle attention mask properly
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Generate text using the model with a specified maximum length
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the generated token IDs into readable text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Run sentiment analysis on the generated response
        sentiment = self.sentiment_pipeline(response)[0]
        
        # Convert sentiment label to a human-readable version
        sentiment_label = self.label_map.get(sentiment["label"], sentiment["label"])
        
        return response, sentiment_label

    # Process a list of prompts and save both detailed and summary results
    def process_prompts(self, prompts, output_path, summary_path):
        sentiments = []  # To store only sentiment labels
        
        # Open file to write full responses and their sentiments
        with open(output_path, "w", encoding="utf-8") as output_file:
            for prompt in prompts:
                # Get the model's response and sentiment for the current prompt
                response, sentiment = self.analyze_prompt(prompt)
                
                # Write everything to the output file
                output_file.write(f"Prompt: {prompt}\nResponse: {response}\nSentiment: {sentiment}\n\n")
                
                # Store the sentiment for summary
                sentiments.append(sentiment)

        # Save just the sentiments in another file (one per line)
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            for sentiment in sentiments:
                summary_file.write(sentiment + "\n")
