import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_toxicity_dataset():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("google/jigsaw_toxicity_pred", data_dir="data", split="train", trust_remote_code=True)
    
    binary_labels = [1 if item['toxic'] >= 0.5 else 0 for item in dataset]
    dataset = dataset.add_column("binary_label", binary_labels)
    
    train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    
    class ToxicityDataset(Dataset):
        def __init__(self, dataset, tokenizer):
            self.dataset = dataset
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            encoding = self.tokenizer(
                item["comment_text"],
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze().to(torch.long),
                "attention_mask": encoding["attention_mask"].squeeze().to(torch.long),
                "label": torch.tensor(item["binary_label"], dtype=torch.long)
            }
    
    train_ds = ToxicityDataset(train_dataset, tokenizer)
    val_ds = ToxicityDataset(val_dataset, tokenizer)
    
    return train_ds, val_ds, tokenizer

def get_dataloaders(train_ds, val_ds, batch_size=16):
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader 