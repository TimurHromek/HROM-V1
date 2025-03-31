import os
# Set parallelism env var *before* importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, disable_caching, concatenate_datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
import math
import re
from datetime import datetime
from contextlib import nullcontext
from collections import defaultdict
import logging
import random # For shuffling combined data

# Disable caching for datasets if needed, helps ensure reprocessing
# disable_caching()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    "dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "ff_dim": 2048,
    "dropout": 0.1,
    "max_seq_len": 512,
    "batch_size": 16,
    "checkpoint_interval": 1500, # Adjusted interval due to more data
    "debug_interval": 300,    # Adjusted interval
    "datasets": ["daily_dialog", "empathetic_dialogues"], # List of datasets
    "tokenizer_name": "hrom_combined_dialog_tokenizer.json", # New tokenizer name
    "checkpoint_dir": "checkpoints_combined_dialog", # New checkpoint dir
    "vocab_size": 32000,
    "tokenizer_train_samples_per_dataset": 100000, # Samples *per dataset* for tokenizer training
    "learning_rate": 3e-5,
    "warmup_steps": 500,
    "max_turns": 8, # Max turns applied per dialogue
    "max_checkpoints": 5,
    "num_epochs": 8,  # Adjusted epochs (more data might require fewer epochs)
    "grad_accum_steps": 8
}


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i, j -> i j", t, self.inv_freq)
        if seq_len == 0:
             return torch.empty((0, self.inv_freq.shape[0] * 2), device=self.inv_freq.device)
        if freqs.shape[0] != seq_len:
             freqs = freqs.reshape(seq_len, -1)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    pos = pos.to(t.device, dtype=t.dtype)
    pos = pos.unsqueeze(0).unsqueeze(1) # Shape: (1, 1, T, dim_rotary)
    tensor_seq_len = t.shape[2]
    pos_seq_len = pos.shape[2]

    if pos_seq_len < tensor_seq_len:
         logging.error(f"RoPE Error: pos sequence length ({pos_seq_len}) is shorter than tensor sequence length ({tensor_seq_len}). Cannot apply.")
         return t
    elif pos_seq_len > tensor_seq_len:
         pos = pos[:, :, :tensor_seq_len, :] # Slice pos to match tensor

    if pos.shape[-1] != t.shape[-1]:
        logging.error(f"Mismatched dimensions for RoPE: pos ({pos.shape[-1]}) vs t ({t.shape[-1]})")
        raise ValueError("Rotary embedding dimension must match head dimension.")

    cos_pos = pos.cos()
    sin_pos = pos.sin()
    rotated_t = (t * cos_pos) + (rotate_half(t) * sin_pos)
    return rotated_t

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)

class HROMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = CONFIG["dim"]
        self.n_heads = CONFIG["n_heads"]
        self.head_dim = self.dim // self.n_heads
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        pos = self.rotary(T)
        q = apply_rotary_pos_emb(pos, q)
        k = apply_rotary_pos_emb(pos, k)
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores + mask # Use provided mask directly
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = attn_probs @ v
        output = output.transpose(1, 2).reshape(B, T, self.dim)
        return self.proj(output)


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
        normed_x = self.norm1(x)
        attn_output = self.attn(normed_x, mask)
        x = x + self.dropout(attn_output)
        normed_x = self.norm2(x)
        ff_output = self.ff(normed_x)
        x = x + self.dropout(ff_output)
        return x

class HROM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG["vocab_size"], CONFIG["dim"])
        self.blocks = nn.ModuleList([HROMBlock() for _ in range(CONFIG["n_layers"])])
        self.norm = nn.LayerNorm(CONFIG["dim"])
        self.head = nn.Linear(CONFIG["dim"], CONFIG["vocab_size"])
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
             torch.nn.init.zeros_(module.bias)
             torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        x = self.dropout(x)

        combined_mask = None
        if attention_mask is not None:
            # Ensure attention_mask is float or compatible type for operations
            attention_mask = attention_mask.to(dtype=torch.float32)
            
            # Causal mask (upper triangular -inf)
            causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.float32) * float('-inf'), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # (1, 1, T, T)

            # Padding mask (from attention_mask: 1s to 0, 0s to -inf)
            # Input attention_mask is (B, T). Convert 0s to large negative, 1s to 0.
            pad_mask_processed = (1.0 - attention_mask) * torch.finfo(torch.float32).min
            pad_mask_processed = pad_mask_processed.unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)

            # Combine: Add the masks. Broadcasting handles the dimensions.
            # (1, 1, T, T) + (B, 1, 1, T) -> (B, 1, T, T)
            combined_mask = causal_mask + pad_mask_processed
            # Ensure mask type matches scores (often float32 or float16 with AMP)
            combined_mask = combined_mask.to(dtype=x.dtype)

        else:
            # No padding mask, just causal mask
             causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device) * float('-inf'), diagonal=1)
             combined_mask = causal_mask.unsqueeze(0).unsqueeze(1) # (1, 1, T, T)
             combined_mask = combined_mask.to(dtype=x.dtype)


        for block in self.blocks:
            x = block(x, combined_mask)

        x = self.norm(x)
        logits = self.head(x)
        return logits

# --- Tokenizer Training ---

class TokenizerTrainer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
        self.tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])
        self.tokenizer_dir = os.path.dirname(self.tokenizer_path)

    def _clean_text(self, text):
        # Basic cleaning: remove extraneous characters, normalize whitespace
        text = str(text) # Ensure text is string
        text = re.sub(r'_comma_', ',', text) # Specific fix for empathetic_dialogues
        text = re.sub(r'[^\w\s.,!?\'\-:;<>"]', '', text) # Keep basic punctuation + quotes
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        return text

    def train(self, dataset_names):
        logging.info("Starting tokenizer training...")
        text_samples = []
        samples_per_dataset = CONFIG['tokenizer_train_samples_per_dataset']

        # --- Process DailyDialog ---
        if "daily_dialog" in dataset_names:
            logging.info(f"Loading daily_dialog for tokenizer training (max {samples_per_dataset} dialogues)...")
            dd_dataset = load_dataset("daily_dialog", split=f"train[:{samples_per_dataset}]")
            logging.info("Processing daily_dialog...")
            for entry in dd_dataset:
                formatted_dialogue = []
                dialogue = entry['dialog'][:CONFIG["max_turns"]] # Apply max turns
                for i, utterance in enumerate(dialogue):
                    role = "<user>" if i % 2 == 0 else "<assistant>"
                    cleaned_utterance = self._clean_text(utterance)
                    formatted_dialogue.append(f"{role} {cleaned_utterance}")
                text_samples.append(" </s> ".join(formatted_dialogue)) # Join turns for training sample

        # --- Process EmpatheticDialogues ---
        if "empathetic_dialogues" in dataset_names:
            logging.info(f"Loading empathetic_dialogues for tokenizer training (max {samples_per_dataset} dialogues)...")
            # Load enough raw entries to likely cover samples_per_dataset *unique* dialogues
            ed_dataset = load_dataset("empathetic_dialogues", split=f"train[:{samples_per_dataset * 5}]") # Load more initially
            logging.info("Processing empathetic_dialogues...")
            conversations = defaultdict(list)
            for entry in ed_dataset:
                 # Group by conv_id, limit total convos processed
                 if len(conversations) >= samples_per_dataset and entry['conv_id'] not in conversations:
                      continue
                 conversations[entry['conv_id']].append(entry['utterance'])

            for conv_id, utterances in conversations.items():
                formatted_dialogue = []
                turns_to_process = utterances[:CONFIG["max_turns"]] # Apply max turns
                for i, utterance in enumerate(turns_to_process):
                    role = "<user>" if i % 2 == 0 else "<assistant>" # Simple alternation
                    cleaned_utterance = self._clean_text(utterance)
                    formatted_dialogue.append(f"{role} {cleaned_utterance}")
                text_samples.append(" </s> ".join(formatted_dialogue)) # Join turns for training sample


        logging.info(f"Total text samples for tokenizer training: {len(text_samples)}")
        if not text_samples:
            raise ValueError("No text samples collected for tokenizer training. Check dataset loading.")

        logging.info(f"Training BPE tokenizer with vocab size {CONFIG['vocab_size']}...")
        trainer = trainers.BpeTrainer(
            vocab_size=CONFIG["vocab_size"],
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True
        )
        self.tokenizer.train_from_iterator(text_samples, trainer=trainer, length=len(text_samples))

        eos_token_id = self.tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            logging.error("</s> token not found in tokenizer vocab after training!")
            # Find *some* valid ID, though this indicates a problem
            eos_token_id = self.tokenizer.token_to_id("<pad>") or 0

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="$A </s>",
            pair="$A </s> $B </s>", # Treat pairs as concatenated dialogues? Check if needed.
            special_tokens=[("</s>", eos_token_id)],
        )

        logging.info(f"Saving tokenizer to {self.tokenizer_path}")
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        self.tokenizer.save(self.tokenizer_path)
        logging.info("Tokenizer training complete.")

    def get_tokenizer(self):
         if not os.path.exists(self.tokenizer_path):
              raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}. Train tokenizer first.")
         tokenizer = Tokenizer.from_file(self.tokenizer_path)
         # Verify special tokens
         for token in self.special_tokens:
              if tokenizer.token_to_id(token) is None:
                   raise ValueError(f"Special token '{token}' not found in loaded tokenizer!")
         return tokenizer

# --- Dataset Loading and Processing ---

class CombinedChatDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.bos_id = self.tokenizer.token_to_id("<s>")
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.assistant_id = self.tokenizer.token_to_id("<assistant>")
        self.max_length = CONFIG["max_seq_len"]
        self._clean_text = TokenizerTrainer()._clean_text # Reuse cleaning function

        self.all_processed_conversations = []

        # --- Process DailyDialog ---
        if "daily_dialog" in CONFIG["datasets"]:
            logging.info("Loading and processing daily_dialog dataset...")
            dd_dataset = load_dataset("daily_dialog", split="train")
            logging.info(f"Processing {len(dd_dataset)} daily_dialog conversations...")
            for entry in dd_dataset:
                conversation = []
                dialogue = entry['dialog'][:CONFIG["max_turns"]] # Apply max turns
                if not dialogue: continue # Skip empty dialogues
                for i, utterance in enumerate(dialogue):
                    role = "<user>" if i % 2 == 0 else "<assistant>"
                    cleaned_text = self._clean_text(utterance)
                    if cleaned_text: # Only add non-empty turns
                        conversation.append({'role': role, 'text': cleaned_text})
                if conversation: # Only add non-empty processed conversations
                    self.all_processed_conversations.append(conversation)

        # --- Process EmpatheticDialogues ---
        if "empathetic_dialogues" in CONFIG["datasets"]:
            logging.info("Loading and processing empathetic_dialogues dataset...")
            ed_dataset = load_dataset("empathetic_dialogues", split="train")
            logging.info("Grouping empathetic_dialogues by conversation ID...")
            conversations_grouped = defaultdict(list)
            # Group by conv_id first, preserving order via utterance_idx
            raw_turns = sorted(list(ed_dataset), key=lambda x: (x['conv_id'], x['utterance_idx']))
            
            last_conv_id = None
            current_conv_turns = []
            for entry in raw_turns:
                conv_id = entry['conv_id']
                # If new conversation starts
                if conv_id != last_conv_id and last_conv_id is not None:
                    if len(current_conv_turns) > CONFIG["max_turns"]:
                         current_conv_turns = current_conv_turns[:CONFIG["max_turns"]] # Apply max turns retrospectively
                    if current_conv_turns: # If the previous conversation wasn't empty
                        self.all_processed_conversations.append(current_conv_turns)
                    current_conv_turns = [] # Reset for new conversation

                # Process current turn
                text = self._clean_text(entry['utterance'])
                if not text: # Skip empty utterances
                     last_conv_id = conv_id # Still update conv_id tracking
                     continue

                # Handle context and roles for empathetic dialogues
                if entry['utterance_idx'] == 1 and entry['context']:
                    context_text = self._clean_text(entry['context'])
                    if context_text:
                        current_conv_turns.append({'role': '<user>', 'text': context_text})
                    # The first actual utterance is then the assistant's response
                    current_conv_turns.append({'role': '<assistant>', 'text': text})
                else:
                    # Determine role by simple alternation within the *current* built turns
                    last_role = current_conv_turns[-1]['role'] if current_conv_turns else None
                    role = '<assistant>' if last_role == '<user>' else '<user>'
                    current_conv_turns.append({'role': role, 'text': text})
                
                last_conv_id = conv_id
            
            # Add the very last conversation from the loop
            if current_conv_turns:
                 if len(current_conv_turns) > CONFIG["max_turns"]:
                     current_conv_turns = current_conv_turns[:CONFIG["max_turns"]]
                 self.all_processed_conversations.append(current_conv_turns)


        logging.info(f"Total processed conversations from all datasets: {len(self.all_processed_conversations)}")
        if not self.all_processed_conversations:
             raise ValueError("No processed conversations were created from any dataset.")

        logging.info("Shuffling combined dataset...")
        random.shuffle(self.all_processed_conversations)


    def __len__(self):
        return len(self.all_processed_conversations)

    def __getitem__(self, idx):
        # Get the pre-processed conversation (list of turn dicts)
        conversation = self.all_processed_conversations[idx]

        formatted_ids = [self.bos_id] # Start with BOS

        for turn in conversation:
            role_id = self.user_id if turn['role'] == '<user>' else self.assistant_id
            utterance_ids = self.tokenizer.encode(turn['text'], add_special_tokens=False).ids

            # Check length before adding: BOS + current_len + role + utterance + EOS <= max_length
            if len(formatted_ids) + 1 + len(utterance_ids) + 1 > self.max_length:
                break # Stop adding turns

            formatted_ids.append(role_id)
            formatted_ids.extend(utterance_ids)
            formatted_ids.append(self.eos_id)

        # Final truncation (safety net, should be rare with the check above)
        formatted_ids = formatted_ids[:self.max_length]

        # Ensure we have at least one token for the label
        if len(formatted_ids) < 2:
             # This is problematic - return None or handle in collate?
             # For now, let's try to return a minimal valid sequence
             logging.warning(f"Found very short sequence (len {len(formatted_ids)}) at index {idx}. Padding.")
             formatted_ids = [self.bos_id, self.eos_id] # Minimal sequence

        input_ids = formatted_ids[:-1]
        labels = formatted_ids[1:]

        # Should not happen with the check above, but safety assertion
        assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    @staticmethod
    def collate_fn(batch):
        # Filter out None items if __getitem__ returned None
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Return None if the whole batch was invalid

        max_len = max(len(item["input_ids"]) for item in batch)

        # Load tokenizer once to get pad_id
        try:
            # Ensure this path matches the saved combined tokenizer
            tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])
            tokenizer = Tokenizer.from_file(tokenizer_path)
            pad_id = tokenizer.token_to_id("<pad>")
            if pad_id is None: raise ValueError("<pad> token not found")
        except Exception as e:
            logging.error(f"Failed to load tokenizer or get pad_id in collate_fn: {e}")
            pad_id = 0 # Fallback

        inputs, labels, masks = [], [], []
        for item in batch:
            input_len = len(item["input_ids"])
            pad_len = max_len - input_len

            inputs.append(item["input_ids"] + [pad_id] * pad_len)
            # Use pad_id for padding labels (CrossEntropyLoss ignores this index)
            labels.append(item["labels"] + [pad_id] * pad_len)
            masks.append([1] * input_len + [0] * pad_len)

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long)
        }

# --- Trainer, Safety Manager, Checkpoint Manager ---
# (HROMTrainer, SafetyManager, CheckpointManager classes remain largely unchanged,
#  but ensure CheckpointManager uses the new CONFIG["checkpoint_dir"])

class HROMTrainer:
    def __init__(self, model, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.model = model.to(self.device)

        self.use_amp = (self.device.type == "cuda" and hasattr(torch.cuda.amp, "GradScaler"))
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        logging.info(f"Automatic Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CONFIG["learning_rate"], # Base LR
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused= (self.device.type == "cuda")
        )
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
             raise ValueError("<pad> token ID not found in tokenizer")

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.base_lr = CONFIG["learning_rate"]
        self.warmup_steps = CONFIG["warmup_steps"]

    def _adjust_learning_rate(self, step):
        if step < self.warmup_steps:
            lr = self.base_lr * (step + 1) / self.warmup_steps
        else:
            lr = self.base_lr # Could add decay here later
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_step(self, batch):
        autocast_context = torch.cuda.amp.autocast(enabled=self.use_amp)
        with autocast_context:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs.view(-1, CONFIG["vocab_size"]), labels.view(-1))
            scaled_loss = loss / CONFIG["grad_accum_steps"]
        if self.use_amp and self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        return loss.item()

    def clip_and_step(self, current_optimizer_step):
         current_lr = self._adjust_learning_rate(current_optimizer_step)
         if self.use_amp and self.scaler:
             self.scaler.unscale_(self.optimizer)
         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
         if self.use_amp and self.scaler:
             self.scaler.step(self.optimizer)
             self.scaler.update()
         else:
             self.optimizer.step()
         self.optimizer.zero_grad(set_to_none=True)
         return current_lr

class SafetyManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.bad_words = ["kill", "murder", "suicide", "hate", "abuse", "violence", "illegal", "harm"] # Expanded slightly
        self.bad_word_ids = []
        logging.info("Initializing safety manager...")
        for word in self.bad_words:
             # Encode without special tokens to get pure word IDs
             ids = tokenizer.encode(word, add_special_tokens=False).ids
             if ids:
                # Store sequences of IDs longer than 1, or single ID if word is one token
                if len(ids) > 0:
                     self.bad_word_ids.append(ids)
                     logging.debug(f"Encoded bad word '{word}' to IDs: {ids}")
                else: # Should not happen with ByteLevel usually
                     logging.warning(f"Bad word '{word}' encoded to empty sequence.")
             else:
                logging.warning(f"Could not encode bad word '{word}' - skipping.")

        self.eos_id = self.tokenizer.token_to_id("</s>")
        if self.eos_id is None:
            logging.error("</s> token ID not found for SafetyManager!")
            self.eos_id = 0 # Fallback

    def contains_sequence(self, tokens, seq):
        if not seq or len(tokens) < len(seq): return False
        for i in range(len(tokens) - len(seq) + 1):
            if tokens[i:i+len(seq)] == seq:
                return True
        return False

    def content_filter(self, text_ids):
        if not isinstance(text_ids, list):
             logging.warning("Content filter received non-list input, attempting encode.")
             text_ids = self.tokenizer.encode(str(text_ids), add_special_tokens=False).ids

        for bad_ids in self.bad_word_ids:
            if self.contains_sequence(text_ids, bad_ids):
                # Optional: decode for logging
                # decoded_word = self.tokenizer.decode(bad_ids)
                # logging.debug(f"Unsafe sequence detected corresponding to {decoded_word}")
                return False # Unsafe
        return True # Safe

    def generate_safely(self, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Encode prompt - important: add_special_tokens=False if prompt already includes role/EOS etc.
        # Assume prompt is like "<user> Hello"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids 
        
        # Add BOS if it's not already part of the prompt structure
        if not input_ids or input_ids[0] != self.tokenizer.token_to_id("<s>"):
             # Check if prompt *starts* with a role token, if so, add BOS before it
             if input_ids and input_ids[0] in [self.tokenizer.token_to_id("<user>"), self.tokenizer.token_to_id("<assistant>")]:
                   input_ids.insert(0, self.tokenizer.token_to_id("<s>"))
             # Or if it's just plain text, add BOS (less likely for chat)
             # else: input_ids.insert(0, self.tokenizer.token_to_id("<s>"))


        generated_ids = list(input_ids)
        logging.debug(f"Starting safe generation with initial IDs: {generated_ids}")

        with torch.no_grad():
            for _ in range(max_new_tokens):
                current_input = torch.tensor([generated_ids]).to(device)
                
                # Generate attention mask for current input
                attention_mask = torch.ones_like(current_input)

                # Get logits using the model's forward method with mask
                logits = self.model(current_input, attention_mask=attention_mask)
                next_token_logits = logits[:, -1, :]

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # Safety Check before adding
                potential_sequence = generated_ids + [next_token_id]
                if not self.content_filter(potential_sequence):
                    logging.warning(f"Potential unsafe token ({next_token_id}) blocked. Stopping generation.")
                    break

                generated_ids.append(next_token_id)

                if next_token_id == self.eos_id:
                    logging.debug("EOS token generated. Stopping generation.")
                    break
        
        self.model.train()
        # Decode the final sequence, skipping special tokens like BOS/EOS if desired
        return self.tokenizer.decode(generated_ids, skip_special_tokens=False) # Keep special tokens for now


    def debug_generation(self, prompt="<user> Hello! How are you today?"): # Removed </s> from default prompt
         logging.info(f"\n--- Debug Generation & Safety Check ---")
         # Ensure the prompt format matches what the model expects (e.g., ending with role token if needed)
         # If the model expects "</s><assistant>" to follow, add it or adjust prompt.
         # Let's assume the model should generate the assistant reply given the user turn.
         
         # Add EOS after the user turn to signal completion before assistant generation
         if not prompt.endswith("</s>"):
             prompt += " </s>" 
         
         response = self.generate_safely(prompt, max_new_tokens=60, temperature=0.7, top_k=50)
         logging.info(f"Prompt: {prompt}")
         logging.info(f"Response: {response}") # Response will include the prompt + generated part
         logging.info("--- End Debug Generation ---\n")


class CheckpointManager:
    def __init__(self):
        # Use checkpoint directory from CONFIG
        self.checkpoint_dir = CONFIG["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoint directory set to: {self.checkpoint_dir}")

    def save(self, model, optimizer, step):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use a consistent naming scheme, maybe include dataset info if needed
        filename = f"hrom_combined_step{step}_{timestamp}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": CONFIG # Save config with checkpoint
        }
        logging.info(f"Saving checkpoint to {path}...")
        try:
            torch.save(state, path)
            logging.info(f"Checkpoint saved successfully at step {step}.")
            self._cleanup_old_checkpoints()
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    def _cleanup_old_checkpoints(self):
        try:
            checkpoints = sorted(
                [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")],
                key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x))
            )
            num_to_delete = len(checkpoints) - CONFIG["max_checkpoints"]
            if num_to_delete > 0:
                logging.info(f"Max checkpoints ({CONFIG['max_checkpoints']}) reached. Removing {num_to_delete} oldest checkpoints.")
                for i in range(num_to_delete):
                    file_to_remove = os.path.join(self.checkpoint_dir, checkpoints[i])
                    try:
                        os.remove(file_to_remove)
                        logging.info(f"Removed old checkpoint: {checkpoints[i]}")
                    except OSError as e:
                        logging.error(f"Error removing checkpoint {file_to_remove}: {e}")
        except Exception as e:
            logging.error(f"Error during checkpoint cleanup: {e}")

    def load_latest(self, model, optimizer):
        try:
            checkpoints = sorted(
                [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")],
                key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)),
                reverse=True
            )
            if not checkpoints:
                logging.info("No checkpoints found to load.")
                return 0 # Start from step 0

            latest_checkpoint_path = os.path.join(self.checkpoint_dir, checkpoints[0])
            logging.info(f"Loading latest checkpoint from: {latest_checkpoint_path}")
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)

            # Compatibility check (optional but good practice)
            loaded_config = checkpoint.get("config", {})
            # Compare critical params like dim, n_layers, vocab_size etc.
            # if loaded_config.get("dim") != CONFIG.get("dim") or ...:
            #     logging.warning("Checkpoint config differs from current config. Loading may be unstable.")

            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint.get('step', 0) + 1

            logging.info(f"Checkpoint loaded successfully. Resuming from optimizer step {start_step}.")
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(map_location)
            return start_step

        except FileNotFoundError:
            logging.info("No checkpoint directory or file found. Starting training from scratch.")
            return 0
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}. Starting training from scratch.")
            # Consider re-initializing model/optimizer here if needed
            return 0

# --- Training Function ---

def train():
    logging.info("Starting HROM training process on combined datasets...")
    logging.info(f"Configuration: {CONFIG}")

    # --- Tokenizer Setup ---
    tokenizer_trainer = TokenizerTrainer()
    tokenizer_path = tokenizer_trainer.tokenizer_path
    if not os.path.exists(tokenizer_path):
        logging.info("Combined tokenizer not found. Training tokenizer...")
        tokenizer_trainer.train(CONFIG["datasets"])
    else:
        logging.info(f"Loading existing combined tokenizer from {tokenizer_path}")
    # Load the tokenizer instance *once* here
    tokenizer = tokenizer_trainer.get_tokenizer()

    # --- Model Initialization ---
    logging.info("Initializing HROM model...")
    model = HROM()

    # --- Dataset and DataLoader ---
    logging.info("Setting up combined dataset and dataloader...")
    try:
         # Ensure datasets are downloaded/cached before creating CombinedChatDataset
         logging.info("Pre-loading/caching datasets...")
         for ds_name in CONFIG["datasets"]:
              logging.info(f"Checking cache for '{ds_name}'...")
              _ = load_dataset(ds_name, split="train", download_mode="reuse_cache_if_exists")
         logging.info("Dataset download/cache check complete.")
         
         # Pass the loaded tokenizer instance
         dataset = CombinedChatDataset(tokenizer)
         
         # Make sure collate_fn doesn't reload tokenizer; it uses CONFIG path now
         dataloader = DataLoader(
             dataset,
             batch_size=CONFIG["batch_size"],
             collate_fn=CombinedChatDataset.collate_fn, # Use static method
             shuffle=True,
             num_workers=4, # Adjust based on your system
             pin_memory=torch.cuda.is_available(),
             prefetch_factor=2 if torch.cuda.is_available() and os.cpu_count() > 1 else None # Optional: speed up loading
         )
    except Exception as e:
         logging.error(f"Failed to initialize dataset/dataloader: {e}", exc_info=True)
         return

    # --- Trainer, Checkpoint, Safety ---
    logging.info("Initializing Trainer, Checkpoint Manager, and Safety Manager...")
    # Pass the loaded tokenizer instance
    trainer_obj = HROMTrainer(model, tokenizer)
    checkpoint_manager = CheckpointManager() # Uses CONFIG["checkpoint_dir"]
    safety = SafetyManager(model, tokenizer) # Pass the loaded tokenizer instance

    # --- Load Checkpoint ---
    start_optimizer_step = checkpoint_manager.load_latest(model, trainer_obj.optimizer)
    model.to(trainer_obj.device)

    # --- Training Loop ---
    logging.info(f"Starting training from optimizer step {start_optimizer_step}")
    optimizer_step = start_optimizer_step
    total_loss_accum = 0.0
    batch_step = 0 # Global batch counter for grad accum

    model.train()

    for epoch in range(CONFIG["num_epochs"]):
        logging.info(f"--- Starting Epoch {epoch+1}/{CONFIG['num_epochs']} ---")

        for i, batch in enumerate(dataloader):
            # Skip batch if collate_fn returned None (e.g., all items invalid)
            if batch is None:
                 logging.warning(f"Skipping empty or invalid batch at step {i} in epoch {epoch+1}")
                 continue

            loss = trainer_obj.train_step(batch)
            if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                 logging.error(f"NaN or Inf loss detected: {loss}. Epoch {epoch+1}, Batch {i}, Opt Step {optimizer_step}. Stopping.")
                 # Consider saving a state here for debugging
                 # checkpoint_manager.save(model, trainer_obj.optimizer, f"{optimizer_step}_nan_inf")
                 return # Stop

            total_loss_accum += loss
            batch_step += 1

            if batch_step % CONFIG["grad_accum_steps"] == 0:
                current_lr = trainer_obj.clip_and_step(optimizer_step)
                avg_loss = total_loss_accum / CONFIG["grad_accum_steps"]
                total_loss_accum = 0.0

                if optimizer_step % CONFIG["debug_interval"] == 0:
                    logging.info(f"Epoch {epoch+1} | Opt Step {optimizer_step} | Batch Step {batch_step} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
                    # Use a consistent debug prompt format
                    safety.debug_generation("<user> How was your day?")

                if optimizer_step > 0 and optimizer_step % CONFIG["checkpoint_interval"] == 0:
                    logging.info(f"Checkpoint interval reached at optimizer step {optimizer_step}.")
                    checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)
                    safety.debug_generation("<user> Can you tell me a joke?")

                optimizer_step += 1

        logging.info(f"--- Finished Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        # Save checkpoint at the end of each epoch
        checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)

    logging.info("Training finished.")
    # Final save
    checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)


if __name__ == "__main__":
    train()
