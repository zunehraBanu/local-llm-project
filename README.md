# Local Language Models - Project 1

## Overview
This project demonstrates the use of two local language models on my machine:
Microsoft Phi-1.5: A generative language model that produces text continuations.
OpenAI GPT-2: A transformer-based generative model used here to generate text based on prompts.

The project reads three predefined prompts from a file (or hard-coded in the scripts), processes them through each model, and saves the outputs in separate text files.

## Prerequisites
## Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.8** (or compatible version)
- **Conda** (Miniconda or Anaconda)
- A stable internet connection for the first run (to download model weights)

## Setup Instructions

### 1. Clone the Repository
Open your terminal or command prompt and run:
<code>git clone https://github.com/yourusername/your-repo-name.git</code>

Then, navigate to the project directory:
<code>cd your-repo-name</code>

### 2. Create and Activate the Conda Environment
Use the provided `requirements.yaml` file to set up the environment. Run:
<code>conda env create -f requirements.yaml</code>

Activate the environment:
<code>conda activate cs_project1</code>

### 3. (Optional) Install Additional Dependencies
All necessary packages are specified in the `requirements.yaml`. If you need to install any additional Python packages, you can use:
<code>pip install package_name</code>

## Running the Models

### 1. Run the Microsoft Phi-1.5 Model
To execute the Phi-1.5 model script, run:
<code>python phi_model.py</code>

This script will:
- Load the Phi-1.5 model and tokenizer.
- Process three predefined prompts.
- Save the generated text responses in `phi_responses.txt`.
- Save the sentiments text responses in `sentiment_summary.txt`.

### 2. Run OpenAI GPT-2 Model
To run the GPT-2 model script, execute:python gpt2_model.py
- This script will:
- Load the GPT-2 model and tokenizer.
- Process the same three prompts.
- Save the generated text responses in `gpt2_responses.txt`.

## File Structure
- **phi_model.py:** Python script to load and run the Microsoft Phi-1.5 model.
- **gpt2_model.py:**  Python script to load and run the OpenAI GPT-2 model.
- **phi_responses.txt:** Output file containing the generated responses from Phi-1.5.
- **sentiment_summary.txt:** Output file containing the the only sentiments from Phi-1.5.
- **gpt2_responses.txt:** Output file containing the generated responses from GPT-2.
- **gpt2_sentiment_summary.txt:** Output file containing the only sentiments from GPT-2.
- **requirements.yaml:** Conda environment file for setting up the necessary dependencies.
- **README.md:** This documentation file.

## Additional Notes
- **Model Downloads:** The first time you run the scripts, the models will be downloaded from Hugging Face. This may take a few minutes depending on your internet speed.
- **Hardware Compatibility:** The scripts are configured to use a GPU if available; otherwise, they will run on the CPU.
- **Troubleshooting:** If you encounter issues, verify that your conda environment is active and all dependencies are installed correctly.
- **Further Reading:** For more details on the models and libraries, please refer to the [Transformers documentation](https://huggingface.co/transformers/) and [PyTorch documentation](https://pytorch.org/).

## All the best! 
