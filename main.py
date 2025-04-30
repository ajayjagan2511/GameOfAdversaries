import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch

from data_prep import prepare_toxicity_dataset, get_dataloaders
from train_safety_head import train_safety_classifier
from safety_oracle import SafetyOracle
from attacker import ContinuousEmbeddingAttacker
from evaluation import verify_attacks, log_attacks, visualize_results

def parse_args():
    parser = argparse.ArgumentParser(description='White-Box Embedding Attack on Mistral-7B Safety Head')
    parser.add_argument('--train', action='store_true', help='Train the safety classifier')
    parser.add_argument('--attack', action='store_true', help='Run attacks')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/safety_head.pt', help='Path to model checkpoint')
    parser.add_argument('--harmful_prompts', type=str, default='./data/harmful_prompts_20000_diverse.csv', help='Path to harmful prompts file')
    parser.add_argument('--max_prompts', type=int, default=1, help='Maximum number of prompts to attack')
    parser.add_argument('--threshold', type=float, default=-1.0, help='Harm probability threshold; <0 uses oracle threshold')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for harm loss')
    parser.add_argument('--lambda_sem', type=float, default=0.1, help='Lambda parameter for semantic loss')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum optimisation steps per prompt')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for attacker optimiser')
    parser.add_argument('--run_id', type=str, default='', help='Optional run identifier for output files')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    suffix = f"_run{args.run_id}" if args.run_id else ""
    attack_file = f"attacks{suffix}.jsonl"
    plots_dir = f"plots{suffix}"
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Train safety classifier if requested
    checkpoint_path = args.checkpoint
    if args.train:
        print("Training safety classifier...")
        checkpoint_path = train_safety_classifier()
        print(f"Safety classifier trained and saved to {checkpoint_path}")
    
    # Run attacks if requested
    if args.attack:
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file {checkpoint_path} not found. Please train the model first.")
            return
        
        # Check if harmful prompts file exists
        if not os.path.exists(args.harmful_prompts):
            print(f"Error: Harmful prompts file {args.harmful_prompts} not found.")
            return
        
        print("Initializing safety oracle...")
        oracle = SafetyOracle(checkpoint_path)
        
        print("Initializing attacker...")
        attacker = ContinuousEmbeddingAttacker(
            oracle, 
            threshold=(None if args.threshold < 0 else args.threshold),
            alpha=args.alpha,
            lambda_sem=args.lambda_sem,
            lr=args.lr,
            max_steps=args.max_steps
        )
        
        print(f"Loading harmful prompts from {args.harmful_prompts}...")
        harmful_prompts = pd.read_csv(args.harmful_prompts)
        
        if len(harmful_prompts) > args.max_prompts:
            harmful_prompts = harmful_prompts.sample(args.max_prompts, random_state=42)
        
        prompts = harmful_prompts['prompt'].tolist()
        
        print(f"Running attacks on {len(prompts)} prompts...")
        results = []
        try:
            for prompt in tqdm(prompts, desc="Attacking prompts"):
                attack_result = attacker.attack(prompt)
                results.append(attack_result)
        except KeyboardInterrupt:
            print("Attack interrupted. Logging partial results...")
        finally:
            print("Verifying attacks using the GPU-loaded oracle...")
            verified_results = verify_attacks(results, oracle)
            print("Logging attack results...")
            if suffix:
                log_attacks(verified_results, output_path=attack_file)
            else:
                log_attacks(verified_results)
            print("Attack phase complete!")
    
    # Visualize results if requested
    if args.visualize:
        print("Visualizing results...")
        if suffix:
            visualize_results(results_path=attack_file, output_dir=plots_dir)
        else:
            visualize_results()
        print("Visualization complete!")

if __name__ == "__main__":
    main() 