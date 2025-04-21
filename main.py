import argparse
import pandas as pd
from safety_oracle import SafetyOracle
from perturbation_attacker import PerturbationAttacker
from logger import Logger


def balanced_sample(df, n):
    if n is None:
        return df
    return df.groupby("category").apply(lambda x: x.sample(min(len(x), n))).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/harmful_prompts_20000_diverse.csv")
    parser.add_argument("--n_per_cat", type=int, default=None)
    parser.add_argument("--max_attacks", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = balanced_sample(df, args.n_per_cat)
    if args.max_attacks:
        df = df.head(args.max_attacks)

    oracle = SafetyOracle()
    attacker = PerturbationAttacker(oracle, threshold=args.threshold)
    logger = Logger()

    for _, row in df.iterrows():
        prompt = row["prompt"]
        csv_score = row.get("harm_score", None)
        record = {"category": row["category"], "csv_harmscore": csv_score}
        attack_result = attacker.attack(prompt)
        record.update(attack_result)
        logger.log(record)


if __name__ == "__main__":
    main() 