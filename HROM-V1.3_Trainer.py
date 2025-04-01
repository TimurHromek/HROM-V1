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
    "checkpoint_interval": 2000,
    "debug_interval": 400,
    # Reverted to training on all four datasets, using correct persona_chat identifier
    "datasets": ["daily_dialog", "empathetic_dialogues", "blended_skill_talk", "AlekseyKorshuk/persona-chat"],
    # Reverted to combined tokenizer name
    "tokenizer_name": "hrom_tokenizer.json",
    # Reverted to combined checkpoint dir
    "checkpoint_dir": "checkpoints",
    "vocab_size": 32000,
    # Adjusted samples per dataset: with 4 datasets, 50k each gives 200k total samples
    "tokenizer_train_samples_per_dataset": 50000,
    "learning_rate": 3e-5,
    "warmup_steps": 500,
    "max_turns": 8, # Max turns applied per dialogue
    "max_checkpoints": 5,
    "num_epochs": 25,
    "grad_accum_steps": 8 # Keep grad accum reasonable
}

# --- Model Definition (HROM, HROMBlock, HROMAttention, SwiGLU, RoPE) ---
# (These classes remain unchanged from the previous version)

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
            elif mask.dim() == 3: # (B, T, T)
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
                # Load more initially to ensure we get enough unique conversations (adjust multiplier if needed)
                ed_dataset = load_dataset("empathetic_dialogues", split=f"train[:{samples_per_dataset * 3}]") # Load *up to* 3x needed
                logging.info("Processing empathetic_dialogues...")
                conversations = defaultdict(list)
                processed_conv_count = 0
                # Group utterances by conv_id first
                grouped_by_conv = defaultdict(list)
                for entry in ed_dataset:
                    grouped_by_conv[entry['conv_id']].append(entry)

                # Process conversations ensuring max samples limit
                for conv_id, entries in grouped_by_conv.items():
                    if processed_conv_count >= samples_per_dataset:
                        break
                    # Sort by utterance_idx to maintain order
                    sorted_entries = sorted(entries, key=lambda x: x['utterance_idx'])
                    formatted_dialogue = []
                    # Handle context and first utterance
                    if sorted_entries[0]['context']:
                         cleaned_context = self._clean_text(sorted_entries[0]['context'])
                         if cleaned_context:
                              formatted_dialogue.append(f"<user> {cleaned_context}") # Assume context is user start
                    # Process subsequent utterances
                    last_role = '<user>' if formatted_dialogue else None # Set initial last role based on context
                    for entry in sorted_entries:
                        cleaned_utterance = self._clean_text(entry['utterance'])
                        if cleaned_utterance:
                            # Determine role based on alternation
                            current_role = '<assistant>' if last_role == '<user>' else '<user>'
                            formatted_dialogue.append(f"{current_role} {cleaned_utterance}")
                            last_role = current_role # Update last role
                    # Apply max turns limit to the formatted turns
                    formatted_dialogue = formatted_dialogue[:CONFIG["max_turns"]]
                    if formatted_dialogue:
                        text_samples.append(" </s> ".join(formatted_dialogue))
                        processed_conv_count += 1 # Count processed unique conversations

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
                    dialogue_turns_raw = entry['previous_utterance']
                    # Add final utterances if they exist and are not empty strings
                    if entry.get('free_turker_utterance'):
                        dialogue_turns_raw.append(entry['free_turker_utterance'])
                    if entry.get('guided_turker_utterance'):
                         dialogue_turns_raw.append(entry['guided_turker_utterance'])

                    turns_to_process = dialogue_turns_raw[:CONFIG["max_turns"]] # Apply max turns limit
                    for i, utterance in enumerate(turns_to_process):
                        role = "<user>" if i % 2 == 0 else "<assistant>" # Assume simple alternation
                        cleaned_utterance = self._clean_text(utterance)
                        if cleaned_utterance:
                            formatted_dialogue.append(f"{role} {cleaned_utterance}")
                    if formatted_dialogue:
                        text_samples.append(" </s> ".join(formatted_dialogue))
            except Exception as e:
                logging.error(f"Failed to load or process blended_skill_talk for tokenizer: {e}")

        # --- Process PersonaChat ---
        if "AlekseyKorshuk/persona-chat" in dataset_names: # Correct dataset identifier
            pc_dataset_name = "AlekseyKorshuk/persona-chat"
            logging.info(f"Loading {pc_dataset_name} for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                pc_dataset = load_dataset(pc_dataset_name, split=f"train[:{samples_per_dataset}]") # Correct dataset identifier
                logging.info(f"Processing {pc_dataset_name}...")
                for entry in pc_dataset:
                    # PersonaChat often has 'utterances' containing 'history'
                    if 'utterances' in entry and entry['utterances']:
                        # Get the history from the last item in utterances for the full dialogue
                        history = entry['utterances'][-1]['history']
                        history = history[:CONFIG["max_turns"]] # Apply max turns
                        formatted_dialogue = []
                        for i, utterance in enumerate(history):
                             role = "<user>" if i % 2 == 0 else "<assistant>" # Assume simple alternation
                             cleaned_utterance = self._clean_text(utterance)
                             if cleaned_utterance:
                                  formatted_dialogue.append(f"{role} {cleaned_utterance}")
                        if formatted_dialogue:
                            text_samples.append(" </s> ".join(formatted_dialogue))
                    else:
                        logging.warning(f"Skipping {pc_dataset_name} entry due to unexpected structure: {entry}")

            except Exception as e:
                logging.error(f"Failed to load or process {pc_dataset_name} for tokenizer: {e}")


        logging.info(f"Total text samples for tokenizer training: {len(text_samples)}")
        if not text_samples:
            raise ValueError("No text samples collected for tokenizer training. Check dataset loading and paths.")

        # Ensure tokenizer directory exists before training
        os.makedirs(self.tokenizer_dir, exist_ok=True)

        logging.info(f"Training BPE tokenizer with vocab size {CONFIG['vocab_size']}...")
        trainer = trainers.BpeTrainer(
            vocab_size=CONFIG["vocab_size"],
            special_tokens=self.special_tokens,
            min_frequency=2, # Keep min_frequency low with more data
            show_progress=True
        )
        # Make sure text_samples is an iterator or list of strings
        def text_iterator():
            for sample in text_samples:
                yield sample

        self.tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=len(text_samples))

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
        self.tokenizer.save(self.tokenizer_path)
        logging.info("Tokenizer training complete.")

    def get_tokenizer(self):
         if not os.path.exists(self.tokenizer_path):
              raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}. Train tokenizer first.")
         tokenizer = Tokenizer.from_file(self.tokenizer_path)
         # Verify special tokens crucial for processing exist
         required_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
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
                conversations_grouped = defaultdict(list)
                for entry in ed_dataset:
                    conversations_grouped[entry['conv_id']].append(entry)

                logging.info(f"Processing {len(conversations_grouped)} empathetic_dialogues conversations...")
                for conv_id, entries in conversations_grouped.items():
                    conversation = []
                    sorted_entries = sorted(entries, key=lambda x: x['utterance_idx'])
                    # Handle context as first user turn if present
                    if sorted_entries[0]['context']:
                        context_text = self._clean_text(sorted_entries[0]['context'])
                        if context_text:
                             conversation.append({'role': '<user>', 'text': context_text})
                    # Process utterances, assuming alternation
                    last_role = conversation[-1]['role'] if conversation else None # Role of the last added turn
                    for entry in sorted_entries:
                         text = self._clean_text(entry['utterance'])
                         if not text: continue
                         # Determine role based on the *last added* role
                         current_role = '<assistant>' if last_role == '<user>' else '<user>'
                         conversation.append({'role': current_role, 'text': text})
                         last_role = current_role # Update for next iteration

                    # Apply max turns limit *after* forming the full sequence
                    conversation = conversation[:CONFIG["max_turns"]]
                    if conversation:
                        self.all_processed_conversations.append(conversation)

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

        # --- Process PersonaChat ---
        if "AlekseyKorshuk/persona-chat" in CONFIG["datasets"]: # Correct dataset identifier
            pc_dataset_name = "AlekseyKorshuk/persona-chat"
            logging.info(f"Loading and processing {pc_dataset_name} dataset...")
            try:
                pc_dataset = load_dataset(pc_dataset_name, split="train") # Correct dataset identifier
                logging.info(f"Processing {len(pc_dataset)} {pc_dataset_name} conversations...")
                for entry in pc_dataset:
                    conversation = []
                    if 'utterances' in entry and entry['utterances']:
                        # Extract the dialogue history
                        history = entry['utterances'][-1]['history']
                        history = history[:CONFIG["max_turns"]] # Apply max turns limit

                        for i, utterance in enumerate(history):
                            role = "<user>" if i % 2 == 0 else "<assistant>" # Simple alternation
                            cleaned_text = self._clean_text(utterance)
                            if cleaned_text:
                                conversation.append({'role': role, 'text': cleaned_text})

                        if conversation: # Only add if not empty
                            self.all_processed_conversations.append(conversation)
                    else:
                         logging.warning(f"Skipping {pc_dataset_name} entry due to unexpected structure: {entry.keys()}")

            except Exception as e:
                logging.error(f"Failed to load or process {pc_dataset_name} for training: {e}")


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
            try:
                utterance_ids = self.tokenizer.encode(turn['text'], add_special_tokens=False).ids
            except Exception as e:
                 logging.error(f"Error encoding text at index {idx}, turn '{turn}': {e}")
                 utterance_ids = [] # Skip this utterance on error

            # Check length: Current + Role + Utterance + EOS <= MaxLength
            # Need +1 for role, +len(utterance), +1 for potential EOS
            if len(formatted_ids) + 1 + len(utterance_ids) + 1 > self.max_length:
                # Attempt to add just the role and EOS if utterance is too long
                if len(formatted_ids) + 1 + 1 <= self.max_length:
                     formatted_ids.append(role_id)
                     formatted_ids.append(self.eos_id)
                break # Stop adding turns

            formatted_ids.append(role_id)
            formatted_ids.extend(utterance_ids)
            formatted_ids.append(self.eos_id)

        # Final safety truncate (should be rare if logic above is correct)
        if len(formatted_ids) > self.max_length:
             formatted_ids = formatted_ids[:self.max_length]
             # Ensure last token isn't partial (though unlikely with BPE)
             # If the truncated sequence ends with a role ID, it's probably bad, remove it.
             if formatted_ids[-1] == self.user_id or formatted_ids[-1] == self.assistant_id:
                  formatted_ids.pop()

        # Handle case of extremely short sequences after processing
        if len(formatted_ids) < 2: # Need at least BOS and one other token for input/label pair
             logging.warning(f"Sequence at index {idx} is too short after processing (<2 tokens). Skipping. Original length: {len(conversation)}")
             # Return None to be filtered by collate_fn
             return None

        input_ids = formatted_ids[:-1]
        labels = formatted_ids[1:]

        # Final check before returning
        if len(input_ids) == 0:
            logging.warning(f"Sequence at index {idx} resulted in empty input_ids after slicing. Skipping.")
            return None


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
            # Correctly reference the tokenizer path from CONFIG within the static method
            tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])
            # TODO: Consider passing tokenizer/pad_id if this becomes a bottleneck
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
            # Pad labels with pad_id (or any ID to be ignored by CrossEntropyLoss)
            labels.append(item["labels"] + [pad_id] * pad_len)
            masks.append([1] * input_len + [0] * pad_len)

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long) # Or bool
        }

# --- Trainer, Safety Manager, Checkpoint Manager ---

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
             # Attempt to get from config if available or fallback
             self.pad_id = CONFIG.get("pad_token_id", 0)
             logging.warning(f"<pad> token ID not found in tokenizer, using fallback ID: {self.pad_id}")


        # Make sure ignore_index uses the determined pad_id
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
        # Determine precision for autocast
        if self.use_amp:
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        autocast_context = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=self.use_amp) if self.use_amp else nullcontext()

        with autocast_context:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask)

            # Reshape for loss calculation
            logits_flat = outputs.view(-1, outputs.size(-1)) # Shape: (B * T, vocab_size)
            labels_flat = labels.view(-1)                   # Shape: (B * T)

            # Calculate loss - ensure logits are float32 for stability esp. with AMP
            loss = self.criterion(logits_flat.float(), labels_flat)

            # Scale loss for gradient accumulation
            scaled_loss = loss / CONFIG["grad_accum_steps"]

        # Backward pass
        if self.use_amp and self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss.item() # Return the unscaled loss for logging

    def clip_and_step(self, current_optimizer_step):
         current_lr = self._adjust_learning_rate(current_optimizer_step)
         # Gradient Clipping *before* optimizer step
         if self.use_amp and self.scaler:
             # Unscale first - important before clipping
             self.scaler.unscale_(self.optimizer)
             # Clip grad norm
             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
             # Optimizer step (with scaler)
             self.scaler.step(self.optimizer)
             # Update scaler for next iteration
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
        # More conservative list
        self.bad_words = ["kill", "murder", "suicide", "hate", "abuse", "violence", "illegal", "harm", "die", "attack", "rape", "molest", "exploit", "terror"]
        self.bad_word_ids = []
        logging.info("Initializing safety manager...")
        # Pre-encode bad word sequences
        for word in self.bad_words:
             # Encode potentially multi-token words carefully
             ids = tokenizer.encode(f" {word}", add_special_tokens=False).ids # Add prefix space for BPE
             if ids:
                self.bad_word_ids.append(ids)
                logging.debug(f"Encoded bad word '{word}' (with space) to IDs: {ids}")
             # Try without space too
             ids_no_space = tokenizer.encode(word, add_special_tokens=False).ids
             if ids_no_space and ids_no_space != ids:
                  self.bad_word_ids.append(ids_no_space)
                  logging.debug(f"Encoded bad word '{word}' (no space) to IDs: {ids_no_space}")

             if not ids and not ids_no_space:
                logging.warning(f"Could not encode bad word '{word}' - skipping.")

        # Pre-get special IDs
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.bos_id = self.tokenizer.token_to_id("<s>")
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.assistant_id = self.tokenizer.token_to_id("<assistant>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")

        if self.eos_id is None: logging.error("</s> token ID not found for SafetyManager!"); self.eos_id = 0
        if self.bos_id is None: logging.error("<s> token ID not found for SafetyManager!"); self.bos_id = 0
        if self.user_id is None: logging.error("<user> token ID not found for SafetyManager!")
        if self.assistant_id is None: logging.error("<assistant> token ID not found for SafetyManager!")
        if self.pad_id is None: logging.error("<pad> token ID not found for SafetyManager!"); self.pad_id = 0


    def contains_sequence(self, tokens, seq):
        """Checks if the list `tokens` contains the sublist `seq`."""
        if not seq or not tokens or len(tokens) < len(seq):
            return False
        seq_len = len(seq)
        for i in range(len(tokens) - seq_len + 1):
            if tokens[i : i + seq_len] == seq:
                return True
        return False

    def content_filter(self, text_ids):
        """Checks if a list of token IDs contains any bad word sequences."""
        if not isinstance(text_ids, list):
            logging.warning("Content filter received non-list input.")
            return True # Default to safe if input is weird
        for bad_ids in self.bad_word_ids:
            if self.contains_sequence(text_ids, bad_ids):
                # Log the detected sequence for debugging
                detected_word = self.tokenizer.decode(bad_ids)
                logging.warning(f"Unsafe content detected: Found sequence corresponding to '{detected_word}' (IDs: {bad_ids}).")
                return False # Unsafe
        return True # Safe

    def generate_safely(self, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        self.model.eval()
        device = next(self.model.parameters()).device

        # Encode prompt, ensure it ends appropriately (e.g., with role token + EOS?)
        # Let's assume the prompt ends like "<user> blah blah </s>" and we need to add "<assistant>"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids

        # Start generation sequence with BOS, prompt, and assistant token
        # Ensure prompt doesn't already include BOS
        if prompt_ids and prompt_ids[0] == self.bos_id:
             input_ids = list(prompt_ids)
        else:
             input_ids = [self.bos_id] + list(prompt_ids)

        # Add the assistant token to signal the model to generate the response
        if self.assistant_id is not None:
            input_ids.append(self.assistant_id)
        else:
            logging.error("Assistant token ID is None, cannot properly start generation.")
            return "Error: Assistant token not found."


        generated_ids = list(input_ids) # Start with the prepared input sequence
        logging.debug(f"Starting safe generation with initial IDs: {generated_ids}")

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Prepare input tensor for this step - only use up to max_seq_len
                current_input_ids = generated_ids[-CONFIG["max_seq_len"]:]
                current_input_tensor = torch.tensor([current_input_ids]).to(device)
                # Create attention mask for the current length
                attention_mask = torch.ones_like(current_input_tensor)

                # Model forward pass
                try:
                    outputs = self.model(current_input_tensor, attention_mask=attention_mask)
                    next_token_logits = outputs[:, -1, :] # Logits for the next token
                except Exception as e:
                     logging.error(f"Model forward pass failed during generation: {e}")
                     break # Stop generation on error

                # --- Safety Check BEFORE sampling ---
                # Apply penalties to bad word starting tokens if possible
                # For now, we filter *after* sampling the token

                # Sampling (Temperature, Top-K)
                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_k > 0 and top_k < next_token_logits.size(-1): # Ensure top_k is valid
                    v, _ = torch.topk(next_token_logits, top_k)
                    # Handle potential NaN/Inf in logits before comparison
                    safe_logits = torch.nan_to_num(next_token_logits, nan=-float('inf'), posinf=float('inf'), neginf=-float('inf'))
                    threshold = v[:, [-1]]
                    safe_logits[safe_logits < threshold] = -float('Inf')
                    next_token_logits = safe_logits # Use the filtered logits

                probs = torch.softmax(next_token_logits, dim=-1)
                # Handle potential NaNs in probabilities before sampling
                if torch.isnan(probs).any():
                     logging.warning("NaN detected in probabilities before sampling. Replacing with uniform distribution.")
                     probs = torch.ones_like(probs) / probs.size(-1) # Fallback to uniform

                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # --- Safety Check AFTER sampling token ---
                # Check if adding this token creates a bad sequence
                potential_sequence_ids = generated_ids + [next_token_id]
                # Check only the newly formed part for bad words for efficiency?
                # Let's check the whole sequence for simplicity/robustness for now.
                if not self.content_filter(potential_sequence_ids):
                    logging.warning(f"Potential unsafe token ({next_token_id}, '{self.tokenizer.decode([next_token_id])}') blocked POST-sampling. Stopping generation.")
                    # Optionally try sampling a different token? For now, just stop.
                    break

                # Add the safe token
                generated_ids.append(next_token_id)

                # Check for EOS token
                if next_token_id == self.eos_id:
                    logging.debug(f"EOS token generated at step {step+1}. Stopping generation.")
                    break

                # Prevent infinite loops if max tokens reached
                if step == max_new_tokens - 1:
                     logging.debug("Max new tokens reached. Stopping generation.")
                     # Ensure the sequence ends with EOS if it didn't naturally
                     if generated_ids[-1] != self.eos_id and self.eos_id is not None:
                         generated_ids.append(self.eos_id)

        self.model.train() # Set model back to training mode

        # Decode the generated part (excluding the initial prompt + assistant token)
        start_index = len(input_ids)
        response_ids = generated_ids[start_index:]

        # Decode, skipping special tokens like EOS, BOS, PAD but potentially keeping USER/ASSISTANT
        # Let's skip all special tokens for the final output text for clarity.
        decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        return decoded_text


    def debug_generation(self, prompt="<user> Tell me about your hobbies."): # Example prompt
         logging.info(f"\n--- Debug Generation & Safety Check ---")
         # Ensure prompt ends logically for the model (e.g., with user token and EOS)
         if not prompt.strip().endswith("</s>"):
              if not prompt.strip().endswith("<user>") and not prompt.strip().endswith("<assistant>"):
                   prompt = prompt.strip() + " </s>" # Add EOS if ends mid-sentence
              else:
                   prompt = prompt.strip() + " </s>" # Add EOS after role token

         # Ensure the prompt starts appropriately (e.g., no BOS needed here as generate_safely adds it)
         if prompt.startswith("<s>"):
              prompt = prompt[len("<s>"):].strip()


         generated_response = self.generate_safely(prompt, max_new_tokens=60, temperature=0.7, top_k=50)

         logging.info(f"Prompt Sent: '{prompt}'")
         logging.info(f"Generated Response: '{generated_response}'")
         logging.info("\n--- End Debug Generation ---\n")


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
        # Ensure step is converted to string if it's passed as something else (e.g., 'final')
        step_str = str(step)
        filename = f"hrom_{prefix}_step{step_str}_{timestamp}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step if isinstance(step, int) else -1, # Store step number or -1 for non-numeric steps
            "config": CONFIG # Save config with checkpoint
        }
        logging.info(f"Saving checkpoint to {path}...")
        try:
            torch.save(state, path)
            logging.info(f"Checkpoint saved successfully at step {step_str}.")
            self._cleanup_old_checkpoints()
        except Exception as e:
            logging.error(f"Failed to save checkpoint '{path}': {e}")

    def _cleanup_old_checkpoints(self):
        max_checkpoints = CONFIG.get("max_checkpoints", 5) # Get from config, default 5
        if max_checkpoints <= 0:
             return # Keep all checkpoints if max_checkpoints is non-positive

        try:
            # Filter only files matching the expected pattern (avoid deleting other files)
            prefix = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
            pattern = re.compile(rf"hrom_{prefix}_step(\d+|.+)_(\d{{8}}_\d{{6}})\.pt")

            checkpoints = []
            for f in os.listdir(self.checkpoint_dir):
                 match = pattern.match(f)
                 if match:
                      filepath = os.path.join(self.checkpoint_dir, f)
                      checkpoints.append((filepath, os.path.getmtime(filepath)))

            # Sort by modification time (oldest first)
            checkpoints.sort(key=lambda x: x[1])

            num_to_delete = len(checkpoints) - max_checkpoints
            if num_to_delete > 0:
                #logging.info(f"Max checkpoints ({max_checkpoints}) reached. Removing {num_to_delete} oldest checkpoints.")
                for i in range(num_to_delete):
                    file_to_remove, _ = checkpoints[i]
                    try:
                        os.remove(file_to_remove)
                        #logging.info(f"Removed old checkpoint: {os.path.basename(file_to_remove)}")
                    except OSError as e:
                        logging.error(f"Error removing checkpoint {file_to_remove}: {e}")
        except Exception as e:
            logging.error(f"Error during checkpoint cleanup: {e}")


    def load_latest(self, model, optimizer):
        try:
            # Filter files based on pattern and sort by time
            prefix = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
            pattern = re.compile(rf"hrom_{prefix}_step(\d+|.+)_(\d{{8}}_\d{{6}})\.pt")
            checkpoints = []
            for f in os.listdir(self.checkpoint_dir):
                 match = pattern.match(f)
                 if match:
                      filepath = os.path.join(self.checkpoint_dir, f)
                      checkpoints.append((filepath, os.path.getmtime(filepath)))

            if not checkpoints:
                logging.info("No valid checkpoints found to load.")
                return 0 # Start from step 0

            # Sort by modification time (newest first)
            checkpoints.sort(key=lambda x: x[1], reverse=True)

            latest_checkpoint_path, _ = checkpoints[0]
            logging.info(f"Loading latest checkpoint from: {latest_checkpoint_path}")
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)

            # --- Config Compatibility Check (Optional but Recommended) ---
            loaded_config = checkpoint.get("config", {})
            # Compare key parameters that affect model architecture or data processing
            critical_keys = ["dim", "n_layers", "n_heads", "ff_dim", "vocab_size", "max_seq_len", "tokenizer_name"]
            mismatched_keys = []
            if loaded_config:
                for key in critical_keys:
                    # Check if key exists in both and if they differ
                    if key in loaded_config and key in CONFIG and loaded_config[key] != CONFIG[key]:
                        mismatched_keys.append((key, loaded_config[key], CONFIG[key]))
                    # Check if key missing in current config but present in checkpoint
                    elif key in loaded_config and key not in CONFIG:
                         mismatched_keys.append((key, loaded_config[key], "Not in current CONFIG"))
                     # Check if key missing in checkpoint config but present in current
                    elif key not in loaded_config and key in CONFIG:
                         mismatched_keys.append((key, "Not in loaded CONFIG", CONFIG[key]))


                if mismatched_keys:
                    logging.warning("--- CONFIG MISMATCH DETECTED ---")
                    logging.warning(f"Checkpoint '{os.path.basename(latest_checkpoint_path)}' was saved with different critical parameters:")
                    for key, loaded_val, current_val in mismatched_keys:
                        logging.warning(f"  - {key}: Checkpoint='{loaded_val}', Current='{current_val}'")
                    # Decide whether to proceed: raise error, warn, or try anyway
                    # For now, just warn strongly. Loading might fail or lead to issues.
                    logging.warning("Proceeding with loading, but results may be unexpected or errors may occur.")
            else:
                logging.warning("Checkpoint does not contain configuration info. Cannot check compatibility.")
            # --- End Config Check ---


            try:
                 # Strict=False can sometimes help load partially, but hides potential issues
                 model.load_state_dict(checkpoint['model'], strict=True)
            except RuntimeError as e:
                 logging.error(f"Failed to load model state_dict: {e}")
                 logging.error("This often happens due to architecture mismatch (check CONFIG) or corrupted checkpoint.")
                 logging.error("Starting training from scratch.")
                 return 0 # Cannot resume if model loading fails

            try:
                 optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                 logging.warning(f"Could not load optimizer state_dict: {e}. Optimizer state will be reset.")
                 # Reinitialize optimizer if state doesn't match? Or just proceed with current state.
                 # Resetting optimizer state is safer if parameters changed.
                 optimizer.state = defaultdict(dict) # Reset state
                 logging.warning("Optimizer state reset.")
            except Exception as e:
                 logging.error(f"Unexpected error loading optimizer state: {e}. Starting training from scratch.")
                 return 0

            start_step = checkpoint.get('step', 0)
            # Ensure step is non-negative, resume from next step
            start_step = max(0, start_step) + 1 if isinstance(start_step, int) else 0


            logging.info(f"Checkpoint loaded successfully. Resuming from optimizer step {start_step}.")
            # Move optimizer state tensors to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        try:
                            state[k] = v.to(map_location)
                        except Exception as e:
                             logging.error(f"Failed to move optimizer tensor '{k}' to device '{map_location}': {e}")
            return start_step

        except FileNotFoundError:
            logging.info(f"No checkpoint directory '{self.checkpoint_dir}' or files found. Starting training from scratch.")
            return 0
        except Exception as e:
            logging.error(f"Error loading checkpoint from '{self.checkpoint_dir}': {e}. Starting training from scratch.")
            # Clean up potentially partially loaded model/optimizer?
            # Re-initializing might be safer depending on where the error occurred.
            # For simplicity, we just return 0 here.
            return 0


# --- Training Function ---

def train():
    logging.info("Starting HROM training process on combined datasets (daily_dialog, empathetic_dialogues, blended_skill_talk, AlekseyKorshuk/persona-chat)...") # Corrected log message
    logging.info(f"Configuration: {CONFIG}")

    # --- Tokenizer Setup ---
    tokenizer_trainer = TokenizerTrainer()
    tokenizer_path = tokenizer_trainer.tokenizer_path
    if not os.path.exists(tokenizer_path):
        logging.info(f"Combined tokenizer '{CONFIG['tokenizer_name']}' not found. Training tokenizer...")
        try:
            tokenizer_trainer.train(CONFIG["datasets"])
        except Exception as e:
             logging.error(f"Failed during tokenizer training: {e}", exc_info=True)
             return # Cannot proceed without a tokenizer
    else:
        logging.info(f"Loading existing combined tokenizer from {tokenizer_path}")
    # Load the tokenizer instance *once* here for shared use
    try:
        tokenizer = tokenizer_trainer.get_tokenizer()
        # Update CONFIG with actual token IDs (useful for downstream)
        CONFIG['pad_token_id'] = tokenizer.token_to_id("<pad>")
        CONFIG['bos_token_id'] = tokenizer.token_to_id("<s>")
        CONFIG['eos_token_id'] = tokenizer.token_to_id("</s>")
        logging.info(f"Loaded tokenizer. Vocab size: {tokenizer.get_vocab_size()}. Special IDs: PAD={CONFIG['pad_token_id']}, BOS={CONFIG['bos_token_id']}, EOS={CONFIG['eos_token_id']}")
    except (FileNotFoundError, ValueError) as e:
         logging.error(f"Failed to load tokenizer: {e}. Cannot continue.")
         return

    # --- Model Initialization ---
    logging.info("Initializing HROM model...")
    # Ensure vocab_size in config matches tokenizer
    if CONFIG['vocab_size'] != tokenizer.get_vocab_size():
         logging.warning(f"Config vocab_size ({CONFIG['vocab_size']}) differs from tokenizer vocab size ({tokenizer.get_vocab_size()}). Using tokenizer's size.")
         CONFIG['vocab_size'] = tokenizer.get_vocab_size()
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

         # Check if dataset is empty after processing
         if len(dataset) == 0:
             logging.error("Dataset is empty after processing all sources. Cannot train.")
             return

         dataloader = DataLoader(
             dataset,
             batch_size=CONFIG["batch_size"],
             collate_fn=CombinedChatDataset.collate_fn, # Use static method
             shuffle=True,
             # Adjust num_workers based on available cores, be conservative
             num_workers=min(4, os.cpu_count() // 2 if (os.cpu_count() and os.cpu_count() > 1) else 1),
             pin_memory=torch.cuda.is_available(),
             prefetch_factor=2 if torch.cuda.is_available() and os.cpu_count() and os.cpu_count() > 1 else None,
             drop_last=False # Keep last batch even if smaller
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
    # Calculate starting batch step based on loaded optimizer step and grad accum
    batch_step = optimizer_step * CONFIG["grad_accum_steps"]
    epochs_completed = batch_step // len(dataloader) if len(dataloader) > 0 else 0
    start_epoch = epochs_completed # Start from the epoch corresponding to the loaded step

    # Estimate total steps (can be useful for LR scheduling if implementing decay)
    try:
        if len(dataloader) == 0:
            raise ValueError("DataLoader has zero length. Cannot estimate total steps.")
        total_optimizer_steps = (len(dataloader) * CONFIG["num_epochs"]) // CONFIG["grad_accum_steps"]
        logging.info(f"Estimated dataset size: {len(dataset)}")
        logging.info(f"Estimated batches per epoch: {len(dataloader)}")
        logging.info(f"Gradient Accumulation Steps: {CONFIG['grad_accum_steps']}")
        logging.info(f"Effective Batch Size: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
        logging.info(f"Target Epochs: {CONFIG['num_epochs']}")
        logging.info(f"Estimated total optimizer steps for {CONFIG['num_epochs']} epochs: {total_optimizer_steps}")
    except Exception as e:
        logging.warning(f"Could not accurately estimate dataloader length or total steps: {e}")
        total_optimizer_steps = -1 # Indicate unknown total steps


    model.train() # Ensure model is in training mode

    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        logging.info(f"--- Starting Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        epoch_loss = 0.0
        num_batches_in_epoch = 0

        # Use enumerate starting from 1 for batch count if preferred
        for i, batch in enumerate(dataloader):
            # Check if batch is valid (collate_fn might return None)
            if batch is None:
                 logging.warning(f"Skipping empty batch at step {i} in epoch {epoch+1}")
                 continue

            # Forward and backward pass (scaled loss)
            loss = trainer_obj.train_step(batch)
            if loss is None or torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                 logging.error(f"NaN, Inf, or None loss detected: {loss}. Epoch {epoch+1}, Batch {i}, Opt Step {optimizer_step}. Stopping.")
                 # Try saving a 'nan_inf' checkpoint before exiting
                 checkpoint_manager.save(model, trainer_obj.optimizer, f"{optimizer_step}_error")
                 return

            total_loss_accum += loss
            epoch_loss += loss
            num_batches_in_epoch += 1
            batch_step += 1 # Increment global batch counter (tracks batches processed)

            # Gradient Accumulation Check & Optimizer Step
            # Check if it's time to perform an optimizer step
            if batch_step % CONFIG["grad_accum_steps"] == 0:
                current_lr = trainer_obj.clip_and_step(optimizer_step) # Pass current opt step for LR schedule

                # Calculate average loss over accumulation steps for logging
                avg_loss = total_loss_accum / CONFIG["grad_accum_steps"]
                total_loss_accum = 0.0 # Reset loss accumulator

                # Logging
                if optimizer_step % CONFIG["debug_interval"] == 0:
                    logging.info(f"Epoch {epoch+1} | Opt Step {optimizer_step} | Batch Step {batch_step} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
                    # Trigger debug generation less frequently or based on condition
                    if optimizer_step % (CONFIG["debug_interval"] * 5) == 0: # e.g., every 5 debug intervals
                         safety.debug_generation("<user> Hi there! How are you doing today?") # Use a generic debug prompt

                # Checkpointing
                if optimizer_step > 0 and optimizer_step % CONFIG["checkpoint_interval"] == 0:
                    logging.info(f"Checkpoint interval reached at optimizer step {optimizer_step}.")
                    checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)
                    # Optional: Run a generation check after saving checkpoint
                    safety.debug_generation("<user> Hi! How are you?")

                optimizer_step += 1 # Increment optimizer step count *after* performing the step

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        logging.info(f"--- Finished Epoch {epoch+1}/{CONFIG['num_epochs']} | Average Epoch Loss: {avg_epoch_loss:.4f} ---")

        # Save checkpoint at the end of each epoch
        checkpoint_manager.save(model, trainer_obj.optimizer, f"epoch{epoch+1}_step{optimizer_step}")
        # Optionally run debug generation at end of epoch
        safety.debug_generation("<user> Hi! Whats up?")


    logging.info(f"Training finished after {CONFIG['num_epochs']} target epochs.")
    # Final save
    logging.info("Saving final model state...")
    checkpoint_manager.save(model, trainer_obj.optimizer, f"final_step{optimizer_step}")


if __name__ == "__main__":
    # Ensures imports happen after setting the env var if script is run directly
    train()
