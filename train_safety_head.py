import os
os.environ['HF_HOME'] = '/scratch/user/aaupadhy/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/user/aaupadhy/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/scratch/user/aaupadhy/.cache/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/scratch/user/aaupadhy/.cache/huggingface'
os.environ['XDG_CACHE_HOME'] = '/scratch/user/aaupadhy/.cache'

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve
import json
from data_prep import prepare_toxicity_dataset, get_dataloaders
import matplotlib.pyplot as plt

class SafetyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.head = nn.Linear(self.base_model.config.hidden_size, 2).to(torch.bfloat16)
        
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        else:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        sequence_output = outputs.hidden_states[-1][:, -1, :]
        logits = self.head(sequence_output)
        return logits
        
    def save(self, path):
        torch.save(self.head.state_dict(), path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        if not ("weight" in state_dict and "bias" in state_dict):
            state_dict = {k[len("head."):]: v for k, v in state_dict.items() if k.startswith("head.")}
        self.head.load_state_dict(state_dict)

def train_safety_classifier():
    if not os.environ.get('HUGGINGFACE_TOKEN'):
        raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable with your Hugging Face token")
    from huggingface_hub import login
    login(token=os.environ['HUGGINGFACE_TOKEN'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds, val_ds, _ = prepare_toxicity_dataset()
    model = SafetyClassifier()
    model.base_model.to(device)

    cache_dir = "./cache_embeddings"
    os.makedirs(cache_dir, exist_ok=True)
    train_emb_file = os.path.join(cache_dir, "train_emb.pt")
    train_label_file = os.path.join(cache_dir, "train_labels.pt")
    val_emb_file = os.path.join(cache_dir, "val_emb.pt")
    val_label_file = os.path.join(cache_dir, "val_labels.pt")

    if not os.path.exists(train_emb_file) or not os.path.exists(val_emb_file):
        tmp_train_loader, tmp_val_loader = get_dataloaders(train_ds, val_ds, batch_size=64)
        model.base_model.eval()
        with torch.no_grad():
            embs, labels = [], []
            for batch in tqdm(tmp_train_loader, desc="Caching train embeddings"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
                seq_out = outputs.hidden_states[-1][:, -1, :]
                embs.append(seq_out.cpu())
                labels.append(batch["label"])
            torch.save(torch.cat(embs, 0), train_emb_file)
            torch.save(torch.cat(labels, 0), train_label_file)
            embs, labels = [], []
            for batch in tqdm(tmp_val_loader, desc="Caching val embeddings"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
                seq_out = outputs.hidden_states[-1][:, -1, :]
                embs.append(seq_out.cpu())
                labels.append(batch["label"])
            torch.save(torch.cat(embs, 0), val_emb_file)
            torch.save(torch.cat(labels, 0), val_label_file)

    class EmbeddingDataset(Dataset):
        def __init__(self, emb_file, label_file):
            self.emb = torch.load(emb_file)
            self.labels = torch.load(label_file)
        def __len__(self):
            return self.emb.size(0)
        def __getitem__(self, idx):
            return self.emb[idx], self.labels[idx]

    train_dataloader = DataLoader(EmbeddingDataset(train_emb_file, train_label_file), batch_size=16, shuffle=True)
    val_dataloader = DataLoader(EmbeddingDataset(val_emb_file, val_label_file), batch_size=16, shuffle=False)

    model.head.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.head.parameters(), lr=5e-5)
    
    # Linear warmup scheduler
    total_steps = len(train_dataloader) * 3  # 3 epochs
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=0.0,
        total_iters=total_steps
    )
    
    best_accuracy = 0.0
    checkpoint_path = "./checkpoints/safety_head.pt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(checkpoint_dir, "safety_head_latest.pt")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt)
        model.head.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        best_accuracy = ckpt.get('best_accuracy', best_accuracy)
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    total_epochs = 3
    
    for epoch in range(start_epoch, total_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]") as pbar:
            for emb, labels in pbar:
                emb = emb.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model.head(emb)
                loss = criterion(logits.float(), labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            with tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]") as pbar:
                for emb, labels in pbar:
                    emb = emb.to(device)
                    labels = labels.to(device)
                    logits = model.head(emb)
                    loss = criterion(logits.float(), labels)
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    val_labels.extend(labels.cpu().tolist())
                    probs = torch.softmax(logits.float(), dim=-1)[:, 1].detach().cpu()
                    val_probs.extend(probs.tolist())
                    accuracy = 100 * correct / total
                    pbar.set_postfix({"loss": loss.item(), "accuracy": accuracy})
        
        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = 100 * correct / total
        
        # Compute ROC-AUC and optimal threshold on the validation split
        try:
            roc_auc = roc_auc_score(val_labels, val_probs)
            fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
            j_scores = tpr - fpr
            best_idx = int(torch.tensor(j_scores).argmax()) if len(j_scores) > 0 else 0
            best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
        except Exception:
            roc_auc = 0.0
            best_threshold = 0.5
        
        print(f"Epoch {epoch+1}/{total_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {accuracy:.2f}%")
        print(f"  ROC-AUC: {roc_auc:.4f} | Optimal threshold: {best_threshold:.4f}")
        
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join("plots", "roc_curve.png"))
        plt.close()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save(checkpoint_path)
            print(f"  Saved new best model with accuracy: {accuracy:.2f}%")

            # Persist the optimal threshold alongside the checkpoint
            with open(os.path.join(checkpoint_dir, "threshold.json"), "w") as fp:
                json.dump({"threshold": best_threshold}, fp)

            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"safety_head_epoch{epoch+1}.pt"))
            torch.save(checkpoint, latest_ckpt)
    
    return checkpoint_path

if __name__ == "__main__":
    train_safety_classifier() 