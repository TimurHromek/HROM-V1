import os
# Set parallelism env var *before* importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Import necessary dataset functions, including concatenate_datasets if needed later
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
    "batch_size": 16, # Keep batch size reasonable
    "checkpoint_interval": 2000, # Increase interval slightly more
    "debug_interval": 400,    # Increase interval slightly more
    "datasets": ["daily_dialog", "empathetic_dialogues", "blended_skill_talk"], # Added blended_skill_talk
    "tokenizer_name": "hrom_bst_combined_tokenizer.json", # Updated tokenizer name
    "checkpoint_dir": "checkpoints_bst_combined", # Updated checkpoint dir
    "vocab_size": 32000,
    "tokenizer_train_samples_per_dataset": 75000, # Adjusted samples per dataset for tokenizer training
    "learning_rate": 3e-5,
    "warmup_steps": 500,
    "max_turns": 8, # Max turns applied per dialogue
    "max_checkpoints": 5,
    "num_epochs": 6,  # Further adjusted epochs (more data)
    "grad_accum_steps": 8 # Keep grad accum reasonable
}

# --- Model Definition (HROM, HROMBlock, HROMAttention, SwiGLU, RoPE) ---


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
        # Defensive reshape only if necessary
        if freqs.shape[0] != seq_len and seq_len > 0:
             freqs = freqs.reshape(seq_len, -1)
        elif seq_len == 0: # Handle edge case for empty sequences
            return torch.empty((0, self.inv_freq.shape[0]*2), device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    # pos: (T, dim_rotary), t: (B, H, T, Head_Dim)
    pos = pos.to(t.device, dtype=t.dtype)
    pos = pos.unsqueeze(0).unsqueeze(1) # Shape: (1, 1, T, dim_rotary)
    tensor_seq_len = t.shape[2]
    pos_seq_len = pos.shape[2]

    if pos_seq_len < tensor_seq_len:
         logging.warning(f"RoPE Warning: pos sequence length ({pos_seq_len}) is shorter than tensor sequence length ({tensor_seq_len}). Using truncated tensor length for RoPE.")
         # This case is tricky, maybe only apply to the length of pos?
         # Or indicates an issue upstream. Let's slice t for now, though it's unusual.
         t_rotated = t[:, :, :pos_seq_len, :]
         pos = pos[:, :, :pos_seq_len, :] # Ensure pos matches the sliced tensor length
         
         # Apply rotation only to the slice
         cos_pos = pos.cos()
         sin_pos = pos.sin()
         t_rotated = (t_rotated * cos_pos) + (rotate_half(t_rotated) * sin_pos)
         
         # Concatenate the rotated part with the un-rotated part
         t_unrotated = t[:, :, pos_seq_len:, :]
         return torch.cat([t_rotated, t_unrotated], dim=2)
         
    elif pos_seq_len > tensor_seq_len:
         pos = pos[:, :, :tensor_seq_len, :] # Slice pos to match tensor

    # Check dimension match after potential slicing
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
        # Generate RoPE embeddings for the current sequence length T
        pos = self.rotary(T) # Shape (T, Head_Dim)
        # Apply RoPE
        q = apply_rotary_pos_emb(pos, q)
        k = apply_rotary_pos_emb(pos, k)
        # Attention calculation
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            # Ensure mask is broadcastable (B, 1, T, T)
            if mask.dim() == 2: # (B, T) -> (B, 1, 1, T) -> add with causal = (B, 1, T, T)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3: # (B, T, T) -> (B, 1, T, T)
                mask = mask.unsqueeze(1)
            # Add mask AFTER scaling scores
            attn_scores = attn_scores + mask # Add large negative values for masked positions
        # Softmax and dropout
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=x.dtype) # Use float for stability
        attn_probs = self.dropout(attn_probs)
        # Output projection
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
        # Pre-Normalization
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
        self.dropout = nn.Dropout(CONFIG["dropout"]) # Add dropout after embedding
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
        x = self.dropout(x) # Apply dropout after embedding

        # Create the combined mask for attention
        combined_mask = None
        # Start with causal mask valid for all sequences in batch
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device) * float('-inf'), diagonal=1)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(1) # (1, 1, T, T)

        if attention_mask is not None:
            # Process padding mask from attention_mask (0 = pad, 1 = real)
            # Convert 0s to -inf, 1s to 0
            pad_mask = (1.0 - attention_mask.to(torch.float32)) * torch.finfo(torch.float32).min
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)
            # Add padding mask to causal mask. Broadcasting ensures (B, 1, T, T)
            # Where pad_mask is -inf, the result is -inf. Otherwise, it's the causal value.
            combined_mask = combined_mask + pad_mask

        # Ensure mask dtype matches data dtype (esp. for AMP)
        combined_mask = combined_mask.to(dtype=x.dtype)

        for block in self.blocks:
            x = block(x, combined_mask) # Pass the combined mask to each block

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
        # Use the updated tokenizer name from CONFIG
        self.tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])
        self.tokenizer_dir = os.path.dirname(self.tokenizer_path)

    def _clean_text(self, text):
        text = str(text) # Ensure text is string
        text = re.sub(r'_comma_', ',', text)
        # Allow alphanumeric, whitespace, and basic punctuation including quotes
        text = re.sub(r'[^\w\s.,!?\'\-:;<>"]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def train(self, dataset_names):
        logging.info("Starting tokenizer training...")
        text_samples = []
        samples_per_dataset = CONFIG['tokenizer_train_samples_per_dataset']

        # --- Process DailyDialog ---
        if "daily_dialog" in dataset_names:
            logging.info(f"Loading daily_dialog for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                # Limit dialogues loaded directly using slicing
                dd_dataset = load_dataset("daily_dialog", split=f"train[:{samples_per_dataset}]")
                logging.info("Processing daily_dialog...")
                for entry in dd_dataset:
                    formatted_dialogue = []
                    dialogue = entry['dialog'][:CONFIG["max_turns"]]
                    for i, utterance in enumerate(dialogue):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_utterance = self._clean_text(utterance)
                        if cleaned_utterance: # Only add non-empty turns
                             formatted_dialogue.append(f"{role} {cleaned_utterance}")
                    if formatted_dialogue: # Only add if dialogue is not empty after cleaning
                        text_samples.append(" </s> ".join(formatted_dialogue))
            except Exception as e:
                logging.error(f"Failed to load or process daily_dialog for tokenizer: {e}")


        # --- Process EmpatheticDialogues ---
        if "empathetic_dialogues" in dataset_names:
            logging.info(f"Loading empathetic_dialogues for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                # Load more initially to ensure we get enough unique conversations
                ed_dataset = load_dataset("empathetic_dialogues", split=f"train[:{samples_per_dataset * 5}]") # Load *up to* 5x needed
                logging.info("Processing empathetic_dialogues...")
                conversations = defaultdict(list)
                processed_conv_count = 0
                for entry in ed_dataset:
                     # Stop collecting if we already have enough unique conversations
                     if processed_conv_count >= samples_per_dataset and entry['conv_id'] not in conversations:
                          continue
                     # Add utterance if we haven't hit the limit or it's part of an ongoing conversation
                     if entry['conv_id'] not in conversations and processed_conv_count < samples_per_dataset:
                          processed_conv_count += 1 # Count unique conversation IDs processed
                     conversations[entry['conv_id']].append(entry['utterance'])

                for conv_id, utterances in conversations.items():
                    formatted_dialogue = []
                    turns_to_process = utterances[:CONFIG["max_turns"]]
                    for i, utterance in enumerate(turns_to_process):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_utterance = self._clean_text(utterance)
                        if cleaned_utterance:
                             formatted_dialogue.append(f"{role} {cleaned_utterance}")
                    if formatted_dialogue:
                         text_samples.append(" </s> ".join(formatted_dialogue))
            except Exception as e:
                logging.error(f"Failed to load or process empathetic_dialogues for tokenizer: {e}")


        # --- Process BlendedSkillTalk ---
        if "blended_skill_talk" in dataset_names:
            logging.info(f"Loading blended_skill_talk for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                # Load dialogues - BST is structured differently, slice directly
                bst_dataset = load_dataset("blended_skill_talk", split=f"train[:{samples_per_dataset}]")
                logging.info("Processing blended_skill_talk...")
                for entry in bst_dataset:
                    formatted_dialogue = []
                    # Combine the dialogue history and the final two turns
                    dialogue_turns = entry['previous_utterance']
                    # Add final utterances if they exist and are not empty strings
                    if entry.get('free_turker_utterance'):
                        dialogue_turns.append(entry['free_turker_utterance'])
                    if entry.get('guided_turker_utterance'):
                         dialogue_turns.append(entry['guided_turker_utterance'])

                    turns_to_process = dialogue_turns[:CONFIG["max_turns"]] # Apply max turns limit
                    for i, utterance in enumerate(turns_to_process):
                        role = "<user>" if i % 2 == 0 else "<assistant>" # Assume simple alternation
                        cleaned_utterance = self._clean_text(utterance)
                        if cleaned_utterance:
                            formatted_dialogue.append(f"{role} {cleaned_utterance}")
                    if formatted_dialogue:
                        text_samples.append(" </s> ".join(formatted_dialogue))
            except Exception as e:
                logging.error(f"Failed to load or process blended_skill_talk for tokenizer: {e}")


        logging.info(f"Total text samples for tokenizer training: {len(text_samples)}")
        if not text_samples:
            raise ValueError("No text samples collected for tokenizer training. Check dataset loading and paths.")

        logging.info(f"Training BPE tokenizer with vocab size {CONFIG['vocab_size']}...")
        trainer = trainers.BpeTrainer(
            vocab_size=CONFIG["vocab_size"],
            special_tokens=self.special_tokens,
            min_frequency=2, # Keep min_frequency low with more data
            show_progress=True
        )
        self.tokenizer.train_from_iterator(text_samples, trainer=trainer, length=len(text_samples))

        eos_token_id = self.tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            logging.warning("</s> token not found in trained tokenizer vocab! Using <pad> as fallback for post-processor.")
            eos_token_id = self.tokenizer.token_to_id("<pad>") or 0 # Fallback needed

        # Configure post-processor (adjust if needed based on how you structure input/output)
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="$A </s>",
            pair="$A </s> $B </s>", # How to handle pairs - maybe just use single always?
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
         # Verify special tokens crucial for processing exist
         required_tokens = ["<pad>", "<s>", "</s>", "<user>", "<assistant>"]
         for token in required_tokens:
              if tokenizer.token_to_id(token) is None:
                   raise ValueError(f"Crucial special token '{token}' not found in loaded tokenizer '{self.tokenizer_path}'!")
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
        # Reuse cleaning function from TokenizerTrainer instance
        self._clean_text = TokenizerTrainer()._clean_text

        self.all_processed_conversations = []

        # --- Process DailyDialog ---
        if "daily_dialog" in CONFIG["datasets"]:
            logging.info("Loading and processing daily_dialog dataset...")
            try:
                dd_dataset = load_dataset("daily_dialog", split="train")
                logging.info(f"Processing {len(dd_dataset)} daily_dialog conversations...")
                for entry in dd_dataset:
                    conversation = []
                    dialogue = entry['dialog'][:CONFIG["max_turns"]]
                    if not dialogue: continue
                    for i, utterance in enumerate(dialogue):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_text = self._clean_text(utterance)
                        if cleaned_text:
                            conversation.append({'role': role, 'text': cleaned_text})
                    if conversation:
                        self.all_processed_conversations.append(conversation)
            except Exception as e:
                 logging.error(f"Failed to load or process daily_dialog for training: {e}")


        # --- Process EmpatheticDialogues ---
        if "empathetic_dialogues" in CONFIG["datasets"]:
            logging.info("Loading and processing empathetic_dialogues dataset...")
            try:
                ed_dataset = load_dataset("empathetic_dialogues", split="train")
                logging.info("Grouping empathetic_dialogues by conversation ID...")
                # Simpler grouping: iterate and build conversations on the fly
                conversations_grouped = defaultdict(list)
                raw_turns = sorted(list(ed_dataset), key=lambda x: (x['conv_id'], x['utterance_idx']))

                last_conv_id = None
                current_conv_turns = []
                for entry in raw_turns:
                    conv_id = entry['conv_id']
                    # If new conversation starts, process the previous one
                    if conv_id != last_conv_id and last_conv_id is not None:
                        if len(current_conv_turns) > CONFIG["max_turns"]:
                            current_conv_turns = current_conv_turns[:CONFIG["max_turns"]]
                        if current_conv_turns:
                            self.all_processed_conversations.append(current_conv_turns)
                        current_conv_turns = [] # Reset

                    # Process current turn
                    text = self._clean_text(entry['utterance'])
                    if not text:
                        last_conv_id = conv_id
                        continue

                    # Handle context and roles for empathetic dialogues
                    if entry['utterance_idx'] == 1 and entry['context']:
                        context_text = self._clean_text(entry['context'])
                        if context_text:
                            current_conv_turns.append({'role': '<user>', 'text': context_text})
                        current_conv_turns.append({'role': '<assistant>', 'text': text})
                    else:
                        last_role = current_conv_turns[-1]['role'] if current_conv_turns else None
                        role = '<assistant>' if last_role == '<user>' else '<user>'
                        current_conv_turns.append({'role': role, 'text': text})
                    last_conv_id = conv_id

                # Add the very last conversation
                if current_conv_turns:
                    if len(current_conv_turns) > CONFIG["max_turns"]:
                        current_conv_turns = current_conv_turns[:CONFIG["max_turns"]]
                    self.all_processed_conversations.append(current_conv_turns)

            except Exception as e:
                logging.error(f"Failed to load or process empathetic_dialogues for training: {e}")


        # --- Process BlendedSkillTalk ---
        if "blended_skill_talk" in CONFIG["datasets"]:
            logging.info("Loading and processing blended_skill_talk dataset...")
            try:
                bst_dataset = load_dataset("blended_skill_talk", split="train")
                logging.info(f"Processing {len(bst_dataset)} blended_skill_talk conversations...")
                for entry in bst_dataset:
                    conversation = []
                    # Reconstruct dialogue: history + final two turns (if they exist)
                    dialogue_turns_raw = entry['previous_utterance']
                    if entry.get('free_turker_utterance'):
                        dialogue_turns_raw.append(entry['free_turker_utterance'])
                    if entry.get('guided_turker_utterance'):
                         dialogue_turns_raw.append(entry['guided_turker_utterance'])

                    if not dialogue_turns_raw: continue # Skip if no turns found

                    turns_to_process = dialogue_turns_raw[:CONFIG["max_turns"]] # Apply max turns limit

                    for i, utterance in enumerate(turns_to_process):
                        role = "<user>" if i % 2 == 0 else "<assistant>" # Assume simple alternation
                        cleaned_text = self._clean_text(utterance)
                        if cleaned_text:
                            conversation.append({'role': role, 'text': cleaned_text})
                    if conversation: # Only add if not empty after cleaning/truncation
                        self.all_processed_conversations.append(conversation)
            except Exception as e:
                logging.error(f"Failed to load or process blended_skill_talk for training: {e}")


        logging.info(f"Total processed conversations from all datasets: {len(self.all_processed_conversations)}")
        if not self.all_processed_conversations:
             raise ValueError("No processed conversations were created from any dataset. Check loading logic and dataset availability.")

        logging.info("Shuffling combined dataset...")
        random.shuffle(self.all_processed_conversations)


    def __len__(self):
        return len(self.all_processed_conversations)

    def __getitem__(self, idx):
        conversation = self.all_processed_conversations[idx]
        formatted_ids = [self.bos_id]
        for turn in conversation:
            role_id = self.user_id if turn['role'] == '<user>' else self.assistant_id
            # Encode without adding special tokens automatically by tokenizer
            utterance_ids = self.tokenizer.encode(turn['text'], add_special_tokens=False).ids

            # Check length: Current + Role + Utterance + EOS <= MaxLength
            if len(formatted_ids) + 1 + len(utterance_ids) + 1 > self.max_length:
                break
            formatted_ids.append(role_id)
            formatted_ids.extend(utterance_ids)
            formatted_ids.append(self.eos_id)

        # Final safety truncate (should be rare)
        formatted_ids = formatted_ids[:self.max_length]

        # Handle case of extremely short sequences after processing
        if len(formatted_ids) < 2:
             logging.warning(f"Sequence at index {idx} is too short after processing (<2 tokens). Skipping. Original: {conversation}")
             # Return None to be filtered by collate_fn
             return None

        input_ids = formatted_ids[:-1]
        labels = formatted_ids[1:]

        assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} vs {len(labels)} at index {idx}"

        return {"input_ids": input_ids, "labels": labels}


    @staticmethod
    def collate_fn(batch):
        # Filter out None items from __getitem__
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Return None if the whole batch was invalid

        max_len = max(len(item["input_ids"]) for item in batch)

        # Load tokenizer once to get pad_id - ensure path matches CONFIG
        try:
            tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])
            # Avoid reloading if already loaded elsewhere, but need pad_id
            # This is slightly inefficient but ensures collate_fn is self-contained if needed
            tokenizer = Tokenizer.from_file(tokenizer_path)
            pad_id = tokenizer.token_to_id("<pad>")
            if pad_id is None: raise ValueError("<pad> token not found")
        except Exception as e:
            logging.error(f"Collate Error: Failed to load tokenizer or get pad_id ('{CONFIG['tokenizer_name']}'): {e}")
            pad_id = 0 # Risky fallback

        inputs, labels, masks = [], [], []
        for item in batch:
            input_len = len(item["input_ids"])
            pad_len = max_len - input_len
            inputs.append(item["input_ids"] + [pad_id] * pad_len)
            # Pad labels with pad_id (ignored by CrossEntropyLoss)
            labels.append(item["labels"] + [pad_id] * pad_len)
            masks.append([1] * input_len + [0] * pad_len)

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long) # Or bool
        }

# --- Trainer, Safety Manager, Checkpoint Manager ---
# (HROMTrainer, SafetyManager, CheckpointManager classes remain largely unchanged,
#  but CheckpointManager now uses the updated CONFIG["checkpoint_dir"])

class HROMTrainer:
    # (No changes needed in HROMTrainer implementation itself)
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
        if self.warmup_steps > 0 and step < self.warmup_steps:
            lr = self.base_lr * (step + 1) / self.warmup_steps
        else:
            # Optional: Add LR decay (e.g., cosine) after warmup
            # Example: lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (total_steps - self.warmup_steps)))
            lr = self.base_lr # Keep base LR after warmup for now
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_step(self, batch):
        autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, enabled=self.use_amp)
        with autocast_context:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # Ensure outputs are float32 for loss calculation if using AMP
            loss = self.criterion(outputs.view(-1, CONFIG["vocab_size"]).float(), labels.view(-1))
            scaled_loss = loss / CONFIG["grad_accum_steps"]
        if self.use_amp and self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        return loss.item()

    def clip_and_step(self, current_optimizer_step):
         current_lr = self._adjust_learning_rate(current_optimizer_step)
         # Gradient Clipping *before* optimizer step
         if self.use_amp and self.scaler:
             # Unscale first
             self.scaler.unscale_(self.optimizer)
             # Clip grad norm
             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
             # Optimizer step
             self.scaler.step(self.optimizer)
             # Update scaler
             self.scaler.update()
         else:
             # Clip grad norm
             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
             # Optimizer step
             self.optimizer.step()

         # Zero gradients *after* stepping
         self.optimizer.zero_grad(set_to_none=True)
         return current_lr


class SafetyManager:
    # (No changes needed in SafetyManager implementation itself)
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.bad_words = ["kill", "murder", "suicide", "hate", "abuse", "violence", "illegal", "harm", "die", "attack"] # Slightly expanded
        self.bad_word_ids = []
        logging.info("Initializing safety manager...")
        # Pre-encode bad word sequences
        for word in self.bad_words:
             ids = tokenizer.encode(word, add_special_tokens=False).ids
             if ids:
                self.bad_word_ids.append(ids)
                logging.debug(f"Encoded bad word '{word}' to IDs: {ids}")
             else:
                logging.warning(f"Could not encode bad word '{word}' - skipping.")
        # Pre-get EOS ID
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.bos_id = self.tokenizer.token_to_id("<s>") # Needed for generation check
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.assistant_id = self.tokenizer.token_to_id("<assistant>")

        if self.eos_id is None:
            logging.error("</s> token ID not found for SafetyManager!")
            self.eos_id = 0 # Fallback

    def contains_sequence(self, tokens, seq):
        if not seq or len(tokens) < len(seq): return False
        seq_len = len(seq)
        for i in range(len(tokens) - seq_len + 1):
            if tokens[i:i+seq_len] == seq:
                return True
        return False

    def content_filter(self, text_ids):
        # Assumes text_ids is list of integers
        if not isinstance(text_ids, list): return True # Cannot filter non-list
        for bad_ids in self.bad_word_ids:
            if self.contains_sequence(text_ids, bad_ids):
                return False # Unsafe
        return True # Safe

    def generate_safely(self, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Encode prompt, don't add special tokens here as structure is controlled below
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        
        # Ensure prompt starts correctly (e.g., with BOS or role token if needed)
        # Assuming standard format is <s> <role> text ... </s>
        if not input_ids or input_ids[0] != self.bos_id:
             # If starts with role, prepend BOS
             if input_ids and input_ids[0] in [self.user_id, self.assistant_id]:
                  input_ids.insert(0, self.bos_id)
             # If just text, prepend BOS and assume user? Or require role in prompt.
             # Let's assume prompt *must* include the role like "<user> Hello"
             # else:
             #     input_ids.insert(0, self.bos_id) # Add BOS if missing

        generated_ids = list(input_ids)
        logging.debug(f"Starting safe generation with initial IDs: {generated_ids}")

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Prepare input tensor for this step
                current_input_tensor = torch.tensor([generated_ids]).to(device)
                # Create attention mask for the current length
                attention_mask = torch.ones_like(current_input_tensor)

                # Model forward pass
                logits = self.model(current_input_tensor, attention_mask=attention_mask)
                next_token_logits = logits[:, -1, :] # Logits for the next token

                # Sampling (Temperature, Top-K)
                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # --- Safety Check BEFORE adding token ---
                potential_sequence = generated_ids + [next_token_id]
                if not self.content_filter(potential_sequence):
                    logging.warning(f"Potential unsafe token ({next_token_id}, '{self.tokenizer.decode([next_token_id])}') blocked. Stopping generation.")
                    break

                generated_ids.append(next_token_id)

                # Check for EOS token
                if next_token_id == self.eos_id:
                    logging.debug("EOS token generated. Stopping generation.")
                    break

        self.model.train() # Set model back to training mode
        # Decode the full generated sequence
        return self.tokenizer.decode(generated_ids, skip_special_tokens=False)


    def debug_generation(self, prompt="<user> What are your hobbies?"): # Example prompt
         logging.info(f"\n--- Debug Generation & Safety Check ---")
         # Ensure prompt ends appropriately to elicit response (e.g., with </s>)
         if not prompt.strip().endswith("</s>"):
             prompt = prompt.strip() + " </s>"

         response = self.generate_safely(prompt, max_new_tokens=60, temperature=0.7, top_k=50)
         logging.info(f"Prompt: {prompt}")
         # The response includes the prompt, find the generated part
         prompt_ids_len = len(self.tokenizer.encode(prompt, add_special_tokens=False).ids)
         response_ids = self.tokenizer.encode(response, add_special_tokens=False).ids
         generated_part_ids = response_ids[prompt_ids_len:]
         generated_part = self.tokenizer.decode(generated_part_ids, skip_special_tokens=True).strip() # Decode only new part

         logging.info(f"Generated: {generated_part}")
         # logging.info(f"Full Response (incl. prompt): {response}") # Optional: log full sequence
         logging.info("--- End Debug Generation ---\n")

class CheckpointManager:
    def __init__(self):
        # Use checkpoint directory from CONFIG
        self.checkpoint_dir = CONFIG["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoint directory set to: {self.checkpoint_dir}")

    def save(self, model, optimizer, step):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use a consistent naming scheme based on the directory name if desired
        prefix = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
        filename = f"hrom_{prefix}_step{step}_{timestamp}.pt"
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
            logging.error(f"Failed to save checkpoint '{path}': {e}")

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

            # Optional: Config compatibility check
            # loaded_config = checkpoint.get("config", {})
            # if loaded_config and loaded_config != CONFIG:
            #     logging.warning("Loaded checkpoint config differs from current CONFIG.")

            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint.get('step', 0) + 1

            logging.info(f"Checkpoint loaded successfully. Resuming from optimizer step {start_step}.")
            # Move optimizer state to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(map_location)
            return start_step

        except FileNotFoundError:
            logging.info(f"No checkpoint directory '{self.checkpoint_dir}' or files found. Starting training from scratch.")
            return 0
        except Exception as e:
            logging.error(f"Error loading checkpoint from '{self.checkpoint_dir}': {e}. Starting training from scratch.")
            return 0


# --- Training Function ---

def train():
    logging.info("Starting HROM training process on combined datasets...")
    logging.info(f"Configuration: {CONFIG}")

    # --- Tokenizer Setup ---
    tokenizer_trainer = TokenizerTrainer()
    tokenizer_path = tokenizer_trainer.tokenizer_path
    if not os.path.exists(tokenizer_path):
        logging.info(f"Combined tokenizer '{CONFIG['tokenizer_name']}' not found. Training tokenizer...")
        tokenizer_trainer.train(CONFIG["datasets"])
    else:
        logging.info(f"Loading existing combined tokenizer from {tokenizer_path}")
    # Load the tokenizer instance *once* here for shared use
    try:
        tokenizer = tokenizer_trainer.get_tokenizer()
    except (FileNotFoundError, ValueError) as e:
         logging.error(f"Failed to load tokenizer: {e}. Cannot continue.")
         return

    # --- Model Initialization ---
    logging.info("Initializing HROM model...")
    model = HROM()

    # --- Dataset and DataLoader ---
    logging.info("Setting up combined dataset and dataloader...")
    try:
         logging.info("Pre-loading/caching datasets...")
         for ds_name in CONFIG["datasets"]:
              logging.info(f"Checking cache for '{ds_name}'...")
              try:
                  # Load just the first example to trigger download/cache check
                  _ = load_dataset(ds_name, split="train[:1]", download_mode="reuse_cache_if_exists")
              except Exception as e:
                  # Log error but try to continue, main dataset loading will handle final error
                  logging.error(f"Could not pre-check dataset '{ds_name}': {e}")
         logging.info("Dataset download/cache check presumed complete.")

         # Pass the already loaded tokenizer instance
         dataset = CombinedChatDataset(tokenizer)

         dataloader = DataLoader(
             dataset,
             batch_size=CONFIG["batch_size"],
             collate_fn=CombinedChatDataset.collate_fn, # Use static method
             shuffle=True,
             num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1), # Safer worker count
             pin_memory=torch.cuda.is_available(),
             prefetch_factor=2 if torch.cuda.is_available() and os.cpu_count() > 1 else None
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
    # Ensure model is on correct device after loading
    model.to(trainer_obj.device)

    # --- Training Loop ---
    logging.info(f"Starting training from optimizer step {start_optimizer_step}")
    optimizer_step = start_optimizer_step
    total_loss_accum = 0.0
    batch_step = optimizer_step * CONFIG["grad_accum_steps"] # Estimate starting batch step

    # Estimate total steps for LR decay (optional)
    try:
        total_optimizer_steps = (len(dataloader) * CONFIG["num_epochs"]) // CONFIG["grad_accum_steps"]
        logging.info(f"Estimated dataset size: {len(dataset)}")
        logging.info(f"Estimated batches per epoch: {len(dataloader)}")
        logging.info(f"Estimated total optimizer steps for {CONFIG['num_epochs']} epochs: {total_optimizer_steps}")
    except Exception as e:
        logging.warning(f"Could not estimate dataset/dataloader length: {e}")
        total_optimizer_steps = -1 # Indicate unknown total steps

    model.train() # Ensure model is in training mode

    for epoch in range(CONFIG["num_epochs"]):
        logging.info(f"--- Starting Epoch {epoch+1}/{CONFIG['num_epochs']} ---")

        for i, batch in enumerate(dataloader):
            if batch is None:
                 logging.warning(f"Skipping empty batch at step {i} in epoch {epoch+1}")
                 continue

            loss = trainer_obj.train_step(batch)
            if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                 logging.error(f"NaN or Inf loss detected: {loss}. Epoch {epoch+1}, Batch {i}, Opt Step {optimizer_step}. Stopping.")
                 checkpoint_manager.save(model, trainer_obj.optimizer, f"{optimizer_step}_nan_inf")
                 return

            total_loss_accum += loss
            batch_step += 1 # Increment global batch counter

            # Gradient Accumulation Check
            if batch_step > 0 and batch_step % CONFIG["grad_accum_steps"] == 0:
                current_lr = trainer_obj.clip_and_step(optimizer_step) # Pass current opt step for LR schedule
                avg_loss = total_loss_accum / CONFIG["grad_accum_steps"]
                total_loss_accum = 0.0 # Reset loss accumulator

                # Logging
                if optimizer_step % CONFIG["debug_interval"] == 0:
                    logging.info(f"Epoch {epoch+1} | Opt Step {optimizer_step} | Batch Step {batch_step} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
                    safety.debug_generation("<user> Tell me something interesting.") # Use a generic debug prompt

                # Checkpointing
                if optimizer_step > 0 and optimizer_step % CONFIG["checkpoint_interval"] == 0:
                    logging.info(f"Checkpoint interval reached at optimizer step {optimizer_step}.")
                    checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)
                    safety.debug_generation("<user> How does empathy work?") # Different debug prompt

                optimizer_step += 1 # Increment optimizer step count *after* performing the step

        logging.info(f"--- Finished Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        # Save checkpoint at the end of each epoch
        checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)

    logging.info("Training finished.")
    # Final save
    checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)


if __name__ == "__main__":
    # Ensures imports happen after setting the env var if script is run directly
    train()
