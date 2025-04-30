# White-Box Embedding Attack on Mistral-7B Safety Head

This project implements a white-box embedding attack on the Mistral-7B language model's safety mechanisms. The implementation follows a comprehensive approach:

1. Fine-tuning a safety classification head on the Jigsaw Toxicity dataset
2. Using that head as a white-box oracle to drive continuous embedding attacks
3. Verifying successful jailbreaks against the real model
4. Logging and visualizing results

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p checkpoints data plots
```

## Usage

The project consists of three main phases:

### 1. Training the Safety Head

```bash
python main.py --train
```

This will:
- Load and process the Jigsaw Toxicity dataset
- Fine-tune a safety classification head on top of Mistral-7B
- Save the checkpoint to `./checkpoints/safety_head.pt`

### 2. Running Attacks

```bash
python main.py --attack --harmful_prompts ./data/harmful_prompts_20000_diverse.csv --max_prompts 100
```

This will:
- Load harmful prompts from the specified CSV file
- Initialize the safety oracle and attacker
- Run embedding attacks on each prompt
- Verify attacks against a freshly loaded oracle
- Log results to `attacks.jsonl`

### 3. Visualizing Results

```bash
python main.py --visualize
```

This will generate plots in the `plots/` directory:
- Distribution of original vs. perturbed harm probabilities
- Distribution of semantic similarities
- Attack success rates

## Parameters

The attacker can be configured with various parameters:

- `--threshold`: Harm probability threshold (default: 0.5)
- `--alpha`: Alpha parameter for harm loss (default: 1.0)
- `--lambda_sem`: Lambda parameter for semantic loss (default: 0.1)
- `--max_prompts`: Maximum number of prompts to attack (default: 100)

## Complete Workflow

To run the entire workflow:

```bash
# Train the safety head
python main.py --train

# Run attacks
python main.py --attack

# Visualize results
python main.py --visualize
```

You can also combine phases:

```bash
python main.py --train --attack --visualize
```