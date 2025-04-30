import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from safety_oracle import SafetyOracle

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#cccccc',
    'axes.spines.top': False,
    'axes.spines.right': False
})

def verify_attacks(attack_results, checkpoint_or_oracle="./checkpoints/safety_head.pt"):
    if isinstance(checkpoint_or_oracle, SafetyOracle):
        oracle = checkpoint_or_oracle
    else:
        oracle = SafetyOracle(checkpoint_or_oracle)
    
    for i, result in enumerate(tqdm(attack_results, desc="Verifying attacks")):
        if result["attack_success"]:
            prob = oracle.predict_proba(result["perturbed_prompt"])
            attack_results[i]["real_world_prob"] = prob
            attack_results[i]["real_world_success"] = prob < oracle.threshold
        else:
            attack_results[i]["real_world_prob"] = result["perturbed_prob"]
            attack_results[i]["real_world_success"] = False
    
    return attack_results

def log_attacks(attack_results, output_path="attacks.jsonl"):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w") as f:
        for result in attack_results:
            f.write(json.dumps(result) + "\n")

def visualize_results(results_path="attacks.jsonl", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"No results file found at {results_path}")
        return
    
    with open(results_path, "r") as f:
        results = [json.loads(line) for line in f]
    
    if not results:
        print("No results to visualize")
        return
    
    original_probs = [r["original_prob"] for r in results]
    perturbed_probs = [r["perturbed_prob"] for r in results]
    similarities = [r["similarity"] for r in results]
    
    # Calculate success rates
    claimed_success = sum(1 for r in results if r["attack_success"]) / len(results)
    real_success = sum(1 for r in results if r.get("real_world_success", False)) / len(results)
    
    # Plot 1: Histogram of original vs. perturbed harm probabilities
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 20)
    plt.hist(original_probs, bins=bins, alpha=0.6, label="Original", color='#1f77b4')
    plt.hist(perturbed_probs, bins=bins, alpha=0.6, label="Perturbed", color='#ff7f0e')
    plt.xlabel("Harm Probability")
    plt.ylabel("Count")
    plt.title("Distribution of Original vs Perturbed Harm Probabilities")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "prob_hist.png"))
    plt.close()
    
    # Plot 2: Distribution of cosine similarities
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, color='skyblue', alpha=0.7)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Distribution of Semantic Similarity Scores")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "sim_dist.png"))
    plt.close()
    
    # Plot 3: Success rate bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Claimed Success", "Real-World Success"], [claimed_success, real_success], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.title("Attack Success Rates")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "success_rate.png"))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")
    print(f"Claimed success rate: {claimed_success:.2%}")
    print(f"Real-world success rate: {real_success:.2%}")

    # Scatter of harm drop vs. semantic similarity
    harm_drops = [o - p for o, p in zip(original_probs, perturbed_probs)]
    plt.figure(figsize=(10, 6))
    plt.scatter(similarities, harm_drops, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Harm Probability Drop")
    plt.title("Harm Probability Drop vs. Semantic Similarity")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "drop_vs_sim.png"))
    plt.close()

    # Threshold Sensitivity Curve
    thresholds = np.linspace(0.3, 0.7, 41)
    success_rates = [(np.sum(np.array(perturbed_probs) < t) / len(perturbed_probs)) for t in thresholds]
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, success_rates, marker="o", color='#2ca02c')
    plt.xlabel("Threshold")
    plt.ylabel("Fraction Below Threshold")
    plt.title("Sensitivity of Attack Success to Threshold")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "threshold_sensitivity.png"))
    plt.close()

    # Prompt Length vs. Attack Success
    lengths = [len(r["original_prompt"].split()) for r in results]
    success_flags = [1 if r.get("attack_success", False) else 0 for r in results]
    plt.figure(figsize=(10, 6))
    jitter = (np.random.rand(len(lengths)) - 0.5) * 0.1
    plt.scatter(lengths, [sf + jitter_i for sf, jitter_i in zip(success_flags, jitter)], alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.xlabel("Prompt Length (words)")
    plt.ylabel("Attack Success (1=Success, 0=Failure)")
    plt.title("Prompt Length vs. Attack Success")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "length_vs_success.png"))
    plt.close()

    # Oracle vs. Real-World Harm Probability
    real_world_probs = [r.get("real_world_prob", None) for r in results]
    plt.figure(figsize=(10, 6))
    plt.scatter(perturbed_probs, real_world_probs, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("Oracle Perturbed Probability")
    plt.ylabel("Real-World Probability")
    plt.title("Oracle vs. Real-World Harm Probabilities")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "oracle_vs_real.png"))
    plt.close()

    # Cumulative Success Over Steps
    max_step = max(r.get("attack_steps", 0) for r in results)
    cumulative = []
    for s in range(1, max_step + 1):
        cum = sum(1 for r in results if r.get("attack_success", False) and r.get("attack_steps", 0) <= s) / len(results)
        cumulative.append(cum)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_step + 1), cumulative, marker='o', color='#9467bd')
    plt.xlabel("Optimization Step")
    plt.ylabel("Cumulative Success Rate")
    plt.title("Cumulative Attack Success Over Steps")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "cumulative_success.png"))
    plt.close()

    # Token-wise Perturbation Heatmap (first 20 prompts)
    if any("delta_norms" in r for r in results):
        delta_matrix = np.array([r["delta_norms"] for r in results[:20]])
        plt.figure(figsize=(12, 6))
        plt.imshow(delta_matrix, aspect="auto", interpolation="nearest", cmap='viridis')
        plt.colorbar(label="ΔE Norm")
        plt.xlabel("Token Position")
        plt.ylabel("Prompt Index")
        plt.title("Token-wise Perturbation Heatmap (first 20 prompts)")
        plt.grid(False)
        plt.savefig(os.path.join(output_dir, "token_perturb_heatmap.png"))
        plt.close()

    # Embedding-space t-SNE Visualization
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.manifold import TSNE
        model = SentenceTransformer('all-MiniLM-L6-v2')
        orig_embs = model.encode([r['original_prompt'] for r in results], convert_to_tensor=False)
        pert_embs = model.encode([r['perturbed_prompt'] for r in results], convert_to_tensor=False)
        all_embs = np.vstack((orig_embs, pert_embs))
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embs)-1))
        tsne_results = tsne.fit_transform(all_embs)
        orig_2d = tsne_results[:len(results)]
        pert_2d = tsne_results[len(results):]
        plt.figure(figsize=(10, 8))
        plt.scatter(orig_2d[:, 0], orig_2d[:, 1], label='Original', alpha=0.6)
        plt.scatter(pert_2d[:, 0], pert_2d[:, 1], label='Perturbed', alpha=0.6)
        plt.legend()
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')
        plt.title('Embedding-space t-SNE')
        plt.savefig(os.path.join(output_dir, 'tsne_embedding.png'))
        plt.close()
    except ImportError:
        pass 