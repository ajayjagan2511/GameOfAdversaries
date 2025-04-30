import torch
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt

class ContinuousEmbeddingAttacker:
    def __init__(self, oracle, threshold=None, alpha=1.0, lambda_sem=0.8, lr=1e-3, max_steps=300, min_similarity=0.9, min_overlap=0.8, lambda_overlap=0.2, lambda_fluency=1e-4):
        self.oracle = oracle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = oracle.tokenizer
        if threshold is None:
            threshold = getattr(oracle, "threshold", 0.5)
        self.threshold = threshold
        self.alpha = alpha
        self.lambda_sem = lambda_sem
        self.lr = lr
        self.max_steps = max_steps
        self.min_similarity = min_similarity
        self.min_overlap = min_overlap
        self.lambda_overlap = lambda_overlap
        self.lambda_fluency = lambda_fluency
        self.excluded_token_ids = {self.tokenizer.pad_token_id}
        if self.tokenizer.eos_token_id is not None:
            self.excluded_token_ids.add(self.tokenizer.eos_token_id)
        bos_id = getattr(self.tokenizer, 'bos_token_id', None)
        if bos_id is not None:
            self.excluded_token_ids.add(bos_id)
        self.sbert_primary = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        try:
            self.sbert_secondary = SentenceTransformer('LaBSE').to(self.device)
        except Exception:
            self.sbert_secondary = None
    
    def embed_text(self, text, secondary=False):
        model = self.sbert_secondary if secondary else self.sbert_primary
        if model is None:
            model = self.sbert_primary
        return model.encode(text, convert_to_tensor=True)
    
    def compute_similarity(self, orig_prompt, pert_prompt):
        orig_emb_1 = self.embed_text(orig_prompt)
        pert_emb_1 = self.embed_text(pert_prompt)
        sim1 = F.cosine_similarity(orig_emb_1.unsqueeze(0), pert_emb_1.unsqueeze(0)).item()
        if self.sbert_secondary is not None:
            orig_emb_2 = self.embed_text(orig_prompt, secondary=True)
            pert_emb_2 = self.embed_text(pert_prompt, secondary=True)
            sim2 = F.cosine_similarity(orig_emb_2.unsqueeze(0), pert_emb_2.unsqueeze(0)).item()
            return min(sim1, sim2)
        return sim1
    
    def token_overlap(self, orig_prompt, pert_prompt):
        orig_tokens = set(orig_prompt.lower().split())
        pert_tokens = set(pert_prompt.lower().split())
        if len(orig_tokens) == 0:
            return 0.0
        return len(orig_tokens & pert_tokens) / len(orig_tokens)
    
    def compute_perplexity(self, prompt):
        tok = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.oracle.model.base_model(**tok, labels=tok["input_ids"])
            ppl = torch.exp(out.loss).item()
        return ppl
    
    def nearest_neighbors_decode(self, embeddings):
        vocab_size = self.tokenizer.vocab_size
        embed_dim = embeddings.size(-1)
        
        all_token_embeds = self.oracle.model.base_model.get_input_embeddings().weight.type_as(embeddings)
        
        batch_size, seq_len, _ = embeddings.shape
        
        decoded_ids = []
        
        for b in range(batch_size):
            sequence_ids = []
            for i in range(seq_len):
                query_embed = embeddings[b, i].unsqueeze(0)
                
                similarities = F.cosine_similarity(
                    query_embed,
                    all_token_embeds
                )
                mask_indices = [idx for idx in self.excluded_token_ids if idx is not None]
                if len(mask_indices) > 0:
                    similarities[mask_indices] = -1e9
                nearest_idx = torch.argmax(similarities).item()
                sequence_ids.append(nearest_idx)
            
            decoded_ids.append(sequence_ids)
        
        return torch.tensor(decoded_ids, device=embeddings.device)
    
    def compute_harm_loss(self, harm_prob):
        if harm_prob > self.threshold:
            return harm_prob
        else:
            return harm_prob - self.alpha * (self.threshold - harm_prob)
    
    def attack(self, prompt):
        tokens = self.oracle.tokenize(prompt)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        
        orig_prob = self.oracle.predict_proba(prompt)
        
        if orig_prob < self.threshold:
            return {
                "original_prompt": prompt,
                "perturbed_prompt": prompt,
                "original_prob": orig_prob,
                "perturbed_prob": orig_prob,
                "similarity": 1.0,
                "attack_steps": 0,
                "attack_success": False
            }
        
        orig_embeds = self.oracle.get_embeddings(input_ids)
        
        # random initialise delta to start near a Voronoi boundary (~0.3 L2-norm per token)
        delta_embeds = torch.randn_like(orig_embeds)
        token_norm = delta_embeds.norm(dim=-1, keepdim=True)
        delta_embeds = 0.3 * delta_embeds / (token_norm + 1e-8)
        delta_embeds.requires_grad_()
        optimizer = optim.Adam([delta_embeds], lr=self.lr)
        
        attack_success = False
        best_prob = orig_prob
        best_delta = delta_embeds.clone().detach()
        steps = []
        harm_probs = []
        sims = []
        losses = []
        plt.ion()
        fig, ax = plt.subplots()
        harm_line, = ax.plot([], [], label='harm_prob')
        sim_line, = ax.plot([], [], label='sim')
        loss_line, = ax.plot([], [], label='loss')
        ax.legend()
        step = 0
        orig_prompt_text = prompt
        try:
            bar = tqdm(range(self.max_steps), desc="Attack Progress")
            for step in bar:
                optimizer.zero_grad()
                
                pert_embeds_cont = orig_embeds + delta_embeds
                
                # Project to nearest tokens to get discrete prompt
                ids_proj = self.nearest_neighbors_decode(pert_embeds_cont)
                pert_embeds_discrete = self.oracle.model.base_model.get_input_embeddings()(ids_proj)
                
                # Straight-through estimator: forward uses discrete, gradient flows through continuous
                pert_embeds_forward = pert_embeds_discrete + (pert_embeds_cont - pert_embeds_cont.detach())
                
                outputs = self.oracle.model.base_model(
                    inputs_embeds=pert_embeds_forward,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                seq_out = outputs.hidden_states[-1][:, -1, :]
                logits = self.oracle.model.head(seq_out)
                probs = torch.softmax(logits, dim=-1)
                harm_prob = probs[:, 1].squeeze()
                
                harm_prob_scalar = harm_prob.item()
                real_len = attention_mask[0].sum().item()
                pert_prompt = self.tokenizer.decode(ids_proj[0][:real_len], skip_special_tokens=True)
                
                sim = self.compute_similarity(prompt, pert_prompt)
                overlap_ratio = self.token_overlap(prompt, pert_prompt)
                perplexity = self.compute_perplexity(pert_prompt)
                
                embed_loss = torch.mean((pert_embeds_cont - orig_embeds) ** 2)
                
                loss_harm = torch.where(harm_prob > self.threshold, harm_prob, harm_prob - self.alpha * (self.threshold - harm_prob))
                loss_overlap_val = 1 - overlap_ratio
                loss_fluency_val = perplexity / 100.0

                optim_loss = (
                    loss_harm
                    + self.lambda_sem * embed_loss
                    + self.lambda_overlap * loss_overlap_val
                )

                # Backpropagate only through differentiable terms
                optim_loss.backward()

                # For logging, include fluency (scaled) but exclude it from gradients
                total_loss_display = optim_loss.detach() + self.lambda_fluency * loss_fluency_val
                
                optimizer.step()
                
                # Clamp delta to keep embeddings within a bounded region
                delta_embeds.data.clamp_(-1.0, 1.0)
                
                bar.set_postfix(harm_prob=f"{harm_prob_scalar:.4f}", sim=f"{sim:.4f}", ov=f"{overlap_ratio:.2f}", loss=f"{total_loss_display.item():.4f}")
                steps.append(step)
                harm_probs.append(harm_prob_scalar)
                sims.append(sim)
                losses.append(total_loss_display.item())
                harm_line.set_data(steps, harm_probs)
                sim_line.set_data(steps, sims)
                loss_line.set_data(steps, losses)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                pert_prob_token = harm_prob_scalar  # already computed on discrete forward pass
                tokens_changed = (ids_proj != input_ids).any().item()
                if (
                    pert_prob_token < self.threshold
                    and sim >= self.min_similarity
                    and overlap_ratio >= self.min_overlap
                    and tokens_changed
                ):
                    attack_success = True
                    best_delta = delta_embeds.clone().detach()
                    break
        except KeyboardInterrupt:
            print(f"Attack interrupted at step {step}")
        
        final_embeds = orig_embeds + best_delta
        decoded_ids = self.nearest_neighbors_decode(final_embeds)
        # Only keep the real tokens for the final prompt
        real_len = attention_mask[0].sum().item()
        pert_prompt = self.tokenizer.decode(decoded_ids[0][:real_len], skip_special_tokens=True)
        
        pert_prob = self.oracle.predict_proba(pert_prompt)
        similarity = self.compute_similarity(prompt, pert_prompt)
        
        if pert_prob < self.threshold:
            attack_success = True
            generation = self.oracle.generate(pert_prompt)
        else:
            generation = ""
        
        delta_norms = best_delta[0].norm(dim=1).cpu().tolist()
        return {
            "original_prompt": prompt,
            "perturbed_prompt": pert_prompt,
            "original_prob": orig_prob,
            "perturbed_prob": pert_prob,
            "similarity": similarity,
            "attack_steps": step + 1 if attack_success else self.max_steps,
            "attack_success": attack_success,
            "generation": generation if attack_success else "",
            "delta_norms": delta_norms
        } 