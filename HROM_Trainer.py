import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
import math
import os
import re
from datetime import datetime
from contextlib import nullcontext

# Configuration
CONFIG = {
    "dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "ff_dim": 2048,
    "dropout": 0.1,
    "max_seq_len": 1024,
    "batch_size": 32,
    "checkpoint_interval": 1000,
    "debug_interval": 500,
    "dataset": "daily_dialog",
    "vocab_size": 32000,
    "tokenizer_train_samples": 100000,
    "learning_rate": 1e-4,  # Lowered learning rate
    "max_turns": 6,
    "max_checkpoints": 5,
    "num_epochs": 100,  # Increased number of epochs
    "grad_accum_steps": 4  # Gradient accumulation steps
}

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i, j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    pos = pos.unsqueeze(0).unsqueeze(1)
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)

class HROMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = CONFIG["dim"]
        self.n_heads = CONFIG["n_heads"]
        self.head_dim = self.dim // self.n_heads
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        pos = self.rotary(T)
        q = apply_rotary_pos_emb(pos, q)
        k = apply_rotary_pos_emb(pos, k)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, self.dim)
        return self.proj(out)

class HROMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = HROMAttention()
        self.ff = nn.Sequential(
            nn.Linear(CONFIG["dim"], 2 * CONFIG["ff_dim"]),
            SwiGLU(),
            nn.Linear(CONFIG["ff_dim"], CONFIG["dim"])
        )
        self.norm1 = nn.LayerNorm(CONFIG["dim"])
        self.norm2 = nn.LayerNorm(CONFIG["dim"])
        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class HROM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG["vocab_size"], CONFIG["dim"])
        self.blocks = nn.ModuleList([HROMBlock() for _ in range(CONFIG["n_layers"])])
        self.norm = nn.LayerNorm(CONFIG["dim"])
        self.head = nn.Linear(CONFIG["dim"], CONFIG["vocab_size"])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, attention_mask=None):
        x = self.embed(x)
        if attention_mask is not None:
            B, T = attention_mask.shape
            causal_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)
            causal_mask = causal_mask.to(x.device)
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.float32)
            pad_mask = (1.0 - pad_mask) * torch.finfo(torch.float32).min
            mask = causal_mask + pad_mask.squeeze(1)
        else:
            B, T = x.shape[:2]
            mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)
            mask = mask.to(x.device)
            mask = mask.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))

class TokenizerTrainer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]

    def train(self, dataset_name):
        dataset = load_dataset(dataset_name, split=f"train[:{CONFIG['tokenizer_train_samples']}]")
        text_samples = []
        for entry in dataset:
            if "dialog" in entry:
                for i, utterance in enumerate(entry["dialog"][:CONFIG["max_turns"]]):
                    role = "<user>" if i % 2 == 0 else "<assistant>"
                    text_samples.append(f"{role} {utterance}")
            else:
                text_samples.append(self._clean_text(entry.get("text", "")))
        trainer = trainers.BpeTrainer(
            vocab_size=CONFIG["vocab_size"],
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True
        )
        self.tokenizer.train_from_iterator(text_samples, trainer=trainer, length=len(text_samples))
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="$A </s>",
            pair="$A $B </s>",
            special_tokens=[("</s>", self.tokenizer.token_to_id("</s>"))],
        )
        os.makedirs("tokenizer", exist_ok=True)
        self.tokenizer.save("tokenizer/hrom_tokenizer.json")

    def _clean_text(self, text):
        text = re.sub(r'[^\w\s.,!?\'\-:;<>]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class ChatDataset(Dataset):
    def __init__(self, tokenizer):
        full_dataset = load_dataset(CONFIG["dataset"], split="train")
        num_samples = min(len(full_dataset), CONFIG["tokenizer_train_samples"])
        self.dataset = full_dataset.shuffle(seed=42).select(range(num_samples))
        self.tokenizer = tokenizer
        self.max_length = CONFIG["max_seq_len"]
        self.turn_sep = self.tokenizer.token_to_id("</s>")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        formatted = []
        if "dialog" in entry:
            dialog = entry["dialog"][:CONFIG["max_turns"]]
            for i, utterance in enumerate(dialog):
                role_token = "<user>" if i % 2 == 0 else "<assistant>"
                formatted.extend([
                    self.tokenizer.token_to_id(role_token),
                    *self.tokenizer.encode(utterance).ids,
                    self.turn_sep
                ])
        else:
            text = entry.get("text", "")
            formatted.extend([
                self.tokenizer.token_to_id("<user>"),
                *self.tokenizer.encode(text).ids,
                self.turn_sep
            ])
        formatted = formatted[:self.max_length-2]
        formatted = [self.tokenizer.token_to_id("<s>"), *formatted, self.tokenizer.token_to_id("</s>")]
        return {
            "input_ids": formatted[:-1],
            "labels": formatted[1:]
        }

    @staticmethod
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        pad_id = Tokenizer.from_file("tokenizer/hrom_tokenizer.json").token_to_id("<pad>")
        inputs, labels, masks = [], [], []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            inputs.append(item["input_ids"] + [pad_id] * pad_len)
            labels.append(item["labels"] + [pad_id] * pad_len)
            masks.append([1] * len(item["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(inputs),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks)
        }

class HROMTrainer:
    def __init__(self, model, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=CONFIG["learning_rate"],
            fused=True if self.device.type == "cuda" else False
        )
        self.tokenizer = tokenizer

    def train_step(self, batch):
        autocast = torch.cuda.amp.autocast if self.device.type == "cuda" else nullcontext
        with autocast():
            outputs = self.model(
                batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device)
            )
            original_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id("<pad>"))(
                outputs.view(-1, CONFIG["vocab_size"]),
                batch["labels"].view(-1).to(self.device)
            )
            scaled_loss = original_loss / CONFIG["grad_accum_steps"]
        
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
            
        return original_loss.item()

    def clip_and_step(self):
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        self.optimizer.zero_grad()

class SafetyManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.bad_words = ["hate", "kill", "harm"]
        self.bad_word_ids = [tokenizer.encode(w).ids for w in self.bad_words]
        
    def content_filter(self, text):
        tokens = self.tokenizer.encode(text).ids
        for bad_ids in self.bad_word_ids:
            if any(tokens[i:i+len(bad_ids)] == bad_ids for i in range(len(tokens))):
                return False
        return True

    def generate_safely(self, prompt, max_length=50):
        input_ids = self.tokenizer.encode(prompt).ids
        device = next(self.model.parameters()).device
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.model(torch.tensor([input_ids]).to(device))
            next_token = logits.argmax(-1)[:, -1].item()
            if next_token == self.tokenizer.token_to_id("</s>"):
                break
            generated = self.tokenizer.decode(input_ids + [next_token])
            if not self.content_filter(generated):
                break
            input_ids.append(next_token)
        return self.tokenizer.decode(input_ids)

    def debug_generation(self, prompt="Hello!"):
        print(f"\nSafety Check Generation:")
        response = self.generate_safely(prompt)
        print(f"Prompt: {prompt}\nResponse: {response}")

class CheckpointManager:
    def __init__(self):
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save(self, model, optimizer, step):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.checkpoint_dir}/hrom_{timestamp}_step{step}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": CONFIG
        }, path)
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(os.listdir(self.checkpoint_dir), 
                             key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)))
        while len(checkpoints) > CONFIG["max_checkpoints"]:
            os.remove(os.path.join(self.checkpoint_dir, checkpoints[0]))
            checkpoints = checkpoints[1:]

def train():
    checkpoint_manager = CheckpointManager()
    if not os.path.exists("tokenizer/hrom_tokenizer.json"):
        print("Training tokenizer...")
        tokenizer_trainer = TokenizerTrainer()
        tokenizer_trainer.train(CONFIG["dataset"])
    
    tokenizer = Tokenizer.from_file("tokenizer/hrom_tokenizer.json")
    model = HROM()
    print("Downloading and caching the dataset...")
    _ = load_dataset(CONFIG["dataset"], split="train", download_mode="reuse_cache_if_exists")
    
    dataset = ChatDataset(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        collate_fn=ChatDataset.collate_fn
    )
    
    trainer_obj = HROMTrainer(model, tokenizer)
    safety = SafetyManager(model, tokenizer)
    
    step = 0
    optimizer_step = 0
    total_loss = 0.0
    model.train()
    
    for epoch in range(CONFIG["num_epochs"]):
        for batch in dataloader:
            loss = trainer_obj.train_step(batch)
            total_loss += loss
            step += 1

            if step % CONFIG["grad_accum_steps"] == 0:
                trainer_obj.clip_and_step()
                avg_loss = total_loss / CONFIG["grad_accum_steps"]
                total_loss = 0.0

                if optimizer_step % CONFIG["checkpoint_interval"] == 0:
                    checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)
                    safety.debug_generation()
                
                if optimizer_step % CONFIG["debug_interval"] == 0:
                    print(f"Optimizer Step {optimizer_step} | Loss: {avg_loss:.4f}")
                    safety.debug_generation("What's the meaning of life?")
                
                optimizer_step += 1

if __name__ == "__main__":
    train()
