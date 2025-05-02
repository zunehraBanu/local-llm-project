import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLMModel:
    def __init__(self):
        self.model_name = "microsoft/phi-1_5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        self.label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs.input_ids, max_length=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment_result = self.sentiment_pipeline(response)[0]
        return self.label_map.get(sentiment_result['label'], sentiment_result['label'])

    def analyze_file(self, input_file='headlines.txt', output_file='sentiments.txt'):
        sentiments = []
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                sentiment = self.analyze_sentiment(line.strip())
                sentiments.append(sentiment)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentiment in sentiments:
                f.write(sentiment + '\n')
