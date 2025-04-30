import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from train_safety_head import SafetyClassifier
import os
import json

class SafetyOracle:
    def __init__(self, checkpoint_path="./checkpoints/safety_head.pt", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = SafetyClassifier()
        self.model.load(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load threshold if available
        thr_path = os.path.join(os.path.dirname(checkpoint_path), "threshold.json")
        if os.path.exists(thr_path):
            try:
                with open(thr_path, "r") as fp:
                    thr_data = json.load(fp)
                self.threshold = float(thr_data.get("threshold", 0.5))
            except Exception:
                self.threshold = 0.5
        else:
            self.threshold = 0.5
    
    def tokenize(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
    
    def get_embeddings(self, input_ids, attention_mask=None):
        with torch.no_grad():
            embeddings = self.model.base_model.get_input_embeddings()(input_ids)
            return embeddings
    
    def predict_proba_from_emb(self, embeddings, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(
                embeddings.size(0), 
                embeddings.size(1), 
                device=embeddings.device,
                dtype=embeddings.dtype
            )
        else:
            attention_mask = attention_mask.to(device=embeddings.device, dtype=embeddings.dtype)
        
        with torch.no_grad():
            outputs = self.model.base_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            last_hidden_state = outputs.hidden_states[-1]
            sequence_output = last_hidden_state[:, -1, :]
            logits = self.model.head(sequence_output)
            probs = F.softmax(logits, dim=-1)
            
            return probs[:, 1].item()
    
    def predict_proba(self, prompt):
        tokens = self.tokenize(prompt)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        
        embeddings = self.get_embeddings(input_ids)
        
        return self.predict_proba_from_emb(embeddings, attention_mask)
    
    def generate(self, prompt, max_new_tokens=128):
        tokens = self.tokenize(prompt)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device).bool()
        
        with torch.no_grad():
            generated_ids = self.model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True) 