from typing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from safety_oracle import SafetyOracle


class PerturbationAttacker:
    def __init__(self, oracle: SafetyOracle, threshold: float = 0.6, alpha: float = 10.0, lambda_: float = 1.0, max_steps: int = 100, lr: float = 1e-3, device: str | None = None):
        self.oracle = oracle
        self.threshold = threshold
        self.alpha = alpha
        self.lambda_ = lambda_
        self.max_steps = max_steps
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = oracle.tokenizer
        self.embeddings = oracle.model.get_input_embeddings()
        sbert_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.sbert_tokenizer = AutoTokenizer.from_pretrained(sbert_name)
        self.sbert_model = AutoModel.from_pretrained(sbert_name).to(self.device)

    def _mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode_sbert(self, text: str):
        encoded = self.sbert_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.sbert_model(**encoded)
        return self._mean_pool(model_output, encoded["attention_mask"])

    def _nearest_tokens(self, embeds: torch.Tensor):
        vocab_embeds = self.embeddings.weight
        logits = torch.matmul(embeds, vocab_embeds.t())
        return logits.argmax(dim=-1)

    def attack(self, prompt: str) -> Dict:
        original_harm, _ = self.oracle.query(prompt)
        original_response = self.oracle.generate_if_jailbroken(prompt, self.threshold)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]
        seq_len = input_ids.shape[1]
        delta = torch.zeros(1, seq_len, self.embeddings.embedding_dim, device=self.device, requires_grad=True)
        optim = torch.optim.Adam([delta], lr=self.lr)
        best_score = original_harm
        best_prompt = prompt
        best_cos = 1.0
        for _ in range(self.max_steps):
            pert_embeds = self.embeddings(input_ids) + delta
            pert_ids = self._nearest_tokens(pert_embeds)
            pert_prompt = self.tokenizer.decode(pert_ids[0], skip_special_tokens=True)
            harm_score, _ = self.oracle.query(pert_prompt)
            if harm_score < best_score:
                best_score = harm_score
                best_prompt = pert_prompt
            orig_emb = self._encode_sbert(prompt)
            pert_emb = self._encode_sbert(pert_prompt)
            cos_sim = F.cosine_similarity(orig_emb, pert_emb).item()
            semantic_loss = 1 - cos_sim
            if harm_score > self.threshold:
                sds_loss = harm_score
            else:
                sds_loss = harm_score - self.alpha * (self.threshold - harm_score)
            loss = torch.tensor(sds_loss, device=self.device) + self.lambda_ * semantic_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            if harm_score < self.threshold:
                break
        perturbed_response = self.oracle.generate_if_jailbroken(prompt, self.threshold) if best_score < self.threshold else ""
        final_cos = best_cos if best_score == original_harm else F.cosine_similarity(self._encode_sbert(prompt), self._encode_sbert(best_prompt)).item()
        return {
            "original_prompt": prompt,
            "perturbed_prompt": best_prompt,
            "original_harmscore": original_harm,
            "perturbed_harmscore": best_score,
            "original_response": original_response,
            "perturbed_response": perturbed_response,
            "cosine_similarity": final_cos,
            "attack_success": best_score < self.threshold,
        } 