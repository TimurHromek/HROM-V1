import os
# Set parallelism env var *before* importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, disable_caching, interleave_datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
import math
import re
from datetime import datetime
from contextlib import nullcontext
from collections import defaultdict
import logging
import random # For shuffling combined data
import itertools # For cycling iterators

# Disable caching for datasets if needed, helps ensure reprocessing
# disable_caching()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Configuration
CONFIG = {
    "dim": 768,
    "n_layers": 8,
    "n_heads": 8,
    "ff_dim": 2048,
    "dropout": 0.1,
    "max_seq_len": 512,
    "batch_size": 16, # Keep batch size reasonable
    "checkpoint_interval": 2000,
    "debug_interval": 400,
    "datasets": [
        "daily_dialog",
        "empathetic_dialogues",
        "blended_skill_talk",
        "AlekseyKorshuk/persona-chat",
        "future-technologies/Universal-Transformers-Dataset" # Added new dataset
    ],
    "tokenizer_name": "hrom_tokenizer.json",
    "checkpoint_dir": "checkpoints",
    "vocab_size": 32000,
    # Adjusted samples per dataset: Skip UT for tokenizer training due to size
    "tokenizer_train_samples_per_dataset": 50000,
    "learning_rate": 2e-5,
    "warmup_steps": 1000,
    "max_turns": 8, # Max turns applied per dialogue (for conversational datasets)
    "max_checkpoints": 5,
    # "num_epochs": 30, # Replaced epochs with max_train_steps for IterableDataset
    "max_train_steps": 150000, # Define total training optimizer steps
    "grad_accum_steps": 8, # Keep grad accum reasonable
    "ut_dataset_config": "code", # Specify which config of Universal Transformers to use (e.g., 'code', 'text')
    "ut_dataset_text_field": "text", # Field containing the text in the UT dataset
    # Define sampling probabilities (optional, sums to 1.0). Adjust as needed.
    # Give conversational datasets higher probability than the generic text/code dataset.
    "dataset_sampling_probabilities": [0.2, 0.2, 0.2, 0.2, 0.2] # Equal for now
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
         t_rotated = t[:, :, :pos_seq_len, :]
         pos = pos[:, :, :pos_seq_len, :]
         cos_pos = pos.cos()
         sin_pos = pos.sin()
         t_rotated = (t_rotated * cos_pos) + (rotate_half(t_rotated) * sin_pos)
         t_unrotated = t[:, :, pos_seq_len:, :]
         return torch.cat([t_rotated, t_unrotated], dim=2)
    elif pos_seq_len > tensor_seq_len:
         pos = pos[:, :, :tensor_seq_len, :]

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
            attn_scores = attn_scores + mask
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=x.dtype)
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
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device) * float('-inf'), diagonal=1)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask.to(torch.float32)) * torch.finfo(torch.float32).min
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = combined_mask + pad_mask

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
        text = str(text)
        text = re.sub(r'_comma_', ',', text)
        text = re.sub(r'[^\w\s.,!?\'\-:;<>"]', '', text) # Keep special tokens <>
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
                dd_dataset = load_dataset("daily_dialog", split=f"train[:{samples_per_dataset}]", trust_remote_code=True)
                logging.info("Processing daily_dialog...")
                for entry in dd_dataset:
                    formatted_dialogue = []
                    dialogue = entry['dialog'][:CONFIG["max_turns"]]
                    for i, utterance in enumerate(dialogue):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_utterance = self._clean_text(utterance)
                        if cleaned_utterance:
                             formatted_dialogue.append(f"{role} {cleaned_utterance}")
                    if formatted_dialogue:
                        text_samples.append(" </s> ".join(formatted_dialogue))
            except Exception as e:
                logging.error(f"Failed to load or process daily_dialog for tokenizer: {e}")

        # --- Process EmpatheticDialogues ---
        if "empathetic_dialogues" in dataset_names:
            logging.info(f"Loading empathetic_dialogues for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                ed_dataset = load_dataset("empathetic_dialogues", split=f"train[:{samples_per_dataset * 3}]", trust_remote_code=True)
                logging.info("Processing empathetic_dialogues...")
                grouped_by_conv = defaultdict(list)
                for entry in ed_dataset:
                    grouped_by_conv[entry['conv_id']].append(entry)

                processed_conv_count = 0
                for conv_id, entries in grouped_by_conv.items():
                    if processed_conv_count >= samples_per_dataset: break
                    sorted_entries = sorted(entries, key=lambda x: x['utterance_idx'])
                    formatted_dialogue = []
                    if sorted_entries[0]['context']:
                         cleaned_context = self._clean_text(sorted_entries[0]['context'])
                         if cleaned_context: formatted_dialogue.append(f"<user> {cleaned_context}")
                    last_role = '<user>' if formatted_dialogue else None
                    for entry in sorted_entries:
                        cleaned_utterance = self._clean_text(entry['utterance'])
                        if cleaned_utterance:
                            current_role = '<assistant>' if last_role == '<user>' else '<user>'
                            formatted_dialogue.append(f"{current_role} {cleaned_utterance}")
                            last_role = current_role
                    formatted_dialogue = formatted_dialogue[:CONFIG["max_turns"]]
                    if formatted_dialogue:
                        text_samples.append(" </s> ".join(formatted_dialogue))
                        processed_conv_count += 1
            except Exception as e:
                logging.error(f"Failed to load or process empathetic_dialogues for tokenizer: {e}")


        # --- Process BlendedSkillTalk ---
        if "blended_skill_talk" in dataset_names:
            logging.info(f"Loading blended_skill_talk for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                bst_dataset = load_dataset("blended_skill_talk", split=f"train[:{samples_per_dataset}]", trust_remote_code=True)
                logging.info("Processing blended_skill_talk...")
                for entry in bst_dataset:
                    formatted_dialogue = []
                    dialogue_turns_raw = entry['previous_utterance']
                    if entry.get('free_turker_utterance'): dialogue_turns_raw.append(entry['free_turker_utterance'])
                    if entry.get('guided_turker_utterance'): dialogue_turns_raw.append(entry['guided_turker_utterance'])
                    turns_to_process = dialogue_turns_raw[:CONFIG["max_turns"]]
                    for i, utterance in enumerate(turns_to_process):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_utterance = self._clean_text(utterance)
                        if cleaned_utterance: formatted_dialogue.append(f"{role} {cleaned_utterance}")
                    if formatted_dialogue:
                        text_samples.append(" </s> ".join(formatted_dialogue))
            except Exception as e:
                logging.error(f"Failed to load or process blended_skill_talk for tokenizer: {e}")

        # --- Process PersonaChat ---
        if "AlekseyKorshuk/persona-chat" in dataset_names:
            pc_dataset_name = "AlekseyKorshuk/persona-chat"
            logging.info(f"Loading {pc_dataset_name} for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                pc_dataset = load_dataset(pc_dataset_name, split=f"train[:{samples_per_dataset}]", trust_remote_code=True)
                logging.info(f"Processing {pc_dataset_name}...")
                for entry in pc_dataset:
                    if 'utterances' in entry and entry['utterances']:
                        history = entry['utterances'][-1]['history'][:CONFIG["max_turns"]]
                        formatted_dialogue = []
                        for i, utterance in enumerate(history):
                             role = "<user>" if i % 2 == 0 else "<assistant>"
                             cleaned_utterance = self._clean_text(utterance)
                             if cleaned_utterance: formatted_dialogue.append(f"{role} {cleaned_utterance}")
                        if formatted_dialogue:
                            text_samples.append(" </s> ".join(formatted_dialogue))
                    else:
                        logging.warning(f"Skipping {pc_dataset_name} entry due to unexpected structure: {entry}")
            except Exception as e:
                logging.error(f"Failed to load or process {pc_dataset_name} for tokenizer: {e}")

        # --- Skip Universal Transformers for Tokenizer Training ---
        if "future-technologies/Universal-Transformers-Dataset" in dataset_names:
             logging.warning("Skipping 'future-technologies/Universal-Transformers-Dataset' for tokenizer training due to its large size and streaming requirement.")
             logging.warning("Tokenizer will be trained only on the conversational datasets.")

        logging.info(f"Total text samples for tokenizer training: {len(text_samples)}")
        if not text_samples:
            raise ValueError("No text samples collected for tokenizer training. Check dataset loading and paths.")

        os.makedirs(self.tokenizer_dir, exist_ok=True)

        logging.info(f"Training BPE tokenizer with vocab size {CONFIG['vocab_size']}...")
        trainer = trainers.BpeTrainer(
            vocab_size=CONFIG["vocab_size"],
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True
        )
        def text_iterator():
            for sample in text_samples: yield sample

        self.tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=len(text_samples))

        eos_token_id = self.tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            logging.warning("</s> token not found! Using <pad> as fallback.")
            eos_token_id = self.tokenizer.token_to_id("<pad>") or 0

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="$A </s>",
            pair="$A </s> $B </s>",
            special_tokens=[("</s>", eos_token_id)],
        )

        logging.info(f"Saving tokenizer to {self.tokenizer_path}")
        self.tokenizer.save(self.tokenizer_path)
        logging.info("Tokenizer training complete.")

    def get_tokenizer(self):
         if not os.path.exists(self.tokenizer_path):
              raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}. Train tokenizer first.")
         tokenizer = Tokenizer.from_file(self.tokenizer_path)
         required_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
         for token in required_tokens:
              if tokenizer.token_to_id(token) is None:
                   raise ValueError(f"Crucial special token '{token}' not found in loaded tokenizer '{self.tokenizer_path}'!")
         return tokenizer

# --- Dataset Loading and Processing (Modified for IterableDataset) ---

class CombinedChatIterableDataset(IterableDataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.bos_id = self.tokenizer.token_to_id("<s>")
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.assistant_id = self.tokenizer.token_to_id("<assistant>")
        self.max_length = CONFIG["max_seq_len"]
        self._clean_text = TokenizerTrainer()._clean_text # Reuse cleaning function

        self.datasets_to_load = CONFIG["datasets"]
        self.sampling_probabilities = CONFIG.get("dataset_sampling_probabilities")
        if self.sampling_probabilities and len(self.sampling_probabilities) != len(self.datasets_to_load):
            logging.warning("Dataset sampling probabilities length mismatch. Using equal probabilities.")
            self.sampling_probabilities = None
        if self.sampling_probabilities and not math.isclose(sum(self.sampling_probabilities), 1.0):
             logging.warning("Dataset sampling probabilities do not sum to 1.0. Normalizing.")
             norm_factor = sum(self.sampling_probabilities)
             self.sampling_probabilities = [p / norm_factor for p in self.sampling_probabilities]


        self.loaded_datasets = []
        self.streamable_datasets = []

        for ds_name in self.datasets_to_load:
            is_streamable = (ds_name == "future-technologies/Universal-Transformers-Dataset")
            logging.info(f"Preparing dataset: {ds_name} (Streaming: {is_streamable})")
            try:
                if is_streamable:
                    # Load the specified config (e.g., 'code') for Universal Transformers Dataset
                    ut_config = CONFIG.get("ut_dataset_config", "text") # Default to 'text' if not specified
                    logging.info(f"Loading Universal Transformers Dataset config: '{ut_config}'")
                    ds = load_dataset(
                        ds_name,
                        ut_config, # Specify the config name here
                        split="train",
                        streaming=True,
                        trust_remote_code=True
                    )
                    # Optional: Apply shuffling with a buffer (can consume memory)
                    # ds = ds.shuffle(buffer_size=10000, seed=42) # Adjust buffer_size as needed
                    self.streamable_datasets.append(ds)
                else:
                    # Load non-streaming datasets fully into memory for shuffling
                    ds = load_dataset(ds_name, split="train", trust_remote_code=True)
                    processed_list = self._process_standard_dataset(ds_name, ds)
                    if processed_list:
                        self.loaded_datasets.append(processed_list)
                    else:
                         logging.warning(f"Dataset {ds_name} resulted in an empty list after processing.")

            except Exception as e:
                 logging.error(f"Failed to load or process dataset '{ds_name}': {e}", exc_info=True)

        if not self.loaded_datasets and not self.streamable_datasets:
            raise ValueError("No datasets could be loaded or processed successfully.")

        logging.info(f"Loaded {len(self.loaded_datasets)} standard datasets and {len(self.streamable_datasets)} streaming datasets.")

        # Calculate total size for shuffling cached datasets
        self.total_cached_items = sum(len(ds_list) for ds_list in self.loaded_datasets)
        logging.info(f"Total items in cached datasets: {self.total_cached_items}")

    def _process_standard_dataset(self, ds_name, dataset):
        """Processes standard conversational datasets into the internal format."""
        processed_conversations = []
        logging.info(f"Processing cached dataset: {ds_name} ({len(dataset)} items)")
        if ds_name == "daily_dialog":
            for entry in dataset:
                conversation = []
                dialogue = entry['dialog'][:CONFIG["max_turns"]]
                if not dialogue: continue
                for i, utterance in enumerate(dialogue):
                    role = "<user>" if i % 2 == 0 else "<assistant>"
                    cleaned_text = self._clean_text(utterance)
                    if cleaned_text: conversation.append({'role': role, 'text': cleaned_text})
                if conversation: processed_conversations.append(conversation)
        elif ds_name == "empathetic_dialogues":
            conversations_grouped = defaultdict(list)
            for entry in dataset: conversations_grouped[entry['conv_id']].append(entry)
            for conv_id, entries in conversations_grouped.items():
                conversation = []
                sorted_entries = sorted(entries, key=lambda x: x['utterance_idx'])
                if sorted_entries[0]['context']:
                    context_text = self._clean_text(sorted_entries[0]['context'])
                    if context_text: conversation.append({'role': '<user>', 'text': context_text})
                last_role = conversation[-1]['role'] if conversation else None
                for entry in sorted_entries:
                     text = self._clean_text(entry['utterance'])
                     if not text: continue
                     current_role = '<assistant>' if last_role == '<user>' else '<user>'
                     conversation.append({'role': current_role, 'text': text})
                     last_role = current_role
                conversation = conversation[:CONFIG["max_turns"]]
                if conversation: processed_conversations.append(conversation)
        elif ds_name == "blended_skill_talk":
            for entry in dataset:
                conversation = []
                dialogue_turns_raw = entry['previous_utterance']
                if entry.get('free_turker_utterance'): dialogue_turns_raw.append(entry['free_turker_utterance'])
                if entry.get('guided_turker_utterance'): dialogue_turns_raw.append(entry['guided_turker_utterance'])
                if not dialogue_turns_raw: continue
                turns_to_process = dialogue_turns_raw[:CONFIG["max_turns"]]
                for i, utterance in enumerate(turns_to_process):
                    role = "<user>" if i % 2 == 0 else "<assistant>"
                    cleaned_text = self._clean_text(utterance)
                    if cleaned_text: conversation.append({'role': role, 'text': cleaned_text})
                if conversation: processed_conversations.append(conversation)
        elif ds_name == "AlekseyKorshuk/persona-chat":
            for entry in dataset:
                conversation = []
                if 'utterances' in entry and entry['utterances']:
                    history = entry['utterances'][-1]['history'][:CONFIG["max_turns"]]
                    for i, utterance in enumerate(history):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_text = self._clean_text(utterance)
                        if cleaned_text: conversation.append({'role': role, 'text': cleaned_text})
                    if conversation: processed_conversations.append(conversation)
                else: logging.warning(f"Skipping {ds_name} entry due to unexpected structure: {entry.keys()}")
        else:
             logging.warning(f"No specific processing logic defined for standard dataset: {ds_name}. Skipping.")
             return [] # Return empty list if no logic matches

        return processed_conversations

    def _tokenize_conversation(self, conversation):
        """Tokenizes a conversation list [{'role': ..., 'text': ...}]"""
        formatted_ids = [self.bos_id]
        for turn in conversation:
            role_id = self.user_id if turn['role'] == '<user>' else self.assistant_id
            try:
                utterance_ids = self.tokenizer.encode(turn['text'], add_special_tokens=False).ids
            except Exception as e:
                 logging.error(f"Error encoding text in turn '{turn}': {e}")
                 utterance_ids = []

            # Check length: Current + Role + Utterance + EOS <= MaxLength
            if len(formatted_ids) + 1 + len(utterance_ids) + 1 > self.max_length:
                if len(formatted_ids) + 1 + 1 <= self.max_length: # Try role + EOS
                     formatted_ids.append(role_id)
                     formatted_ids.append(self.eos_id)
                break # Stop adding turns

            formatted_ids.append(role_id)
            formatted_ids.extend(utterance_ids)
            formatted_ids.append(self.eos_id) # Add EOS after each turn's text

        # Final truncate/cleanup
        if len(formatted_ids) > self.max_length:
             formatted_ids = formatted_ids[:self.max_length]
             if formatted_ids and (formatted_ids[-1] == self.user_id or formatted_ids[-1] == self.assistant_id):
                  formatted_ids.pop()
             if len(formatted_ids) > self.max_length:
                 formatted_ids = formatted_ids[:self.max_length]

        # Ensure sequence is valid for producing labels
        if len(formatted_ids) < 2: return None

        input_ids = formatted_ids[:-1]
        labels = formatted_ids[1:]

        if not input_ids: return None

        return {"input_ids": input_ids, "labels": labels}


    def _tokenize_stream_item(self, item):
        """Tokenizes a single item from the streaming Universal Transformers Dataset."""
        text_field = CONFIG.get("ut_dataset_text_field", "text")
        raw_text = item.get(text_field)
        if not raw_text or not isinstance(raw_text, str):
            logging.debug(f"Skipping stream item due to missing or invalid text field ('{text_field}'). Keys: {item.keys()}")
            return None

        cleaned_text = self._clean_text(raw_text)
        if not cleaned_text:
             logging.debug("Skipping stream item due to empty text after cleaning.")
             return None

        # Treat the entire text block as a single assistant turn for simplicity
        # Format: <s> <assistant> [cleaned text] </s>
        formatted_ids = [self.bos_id, self.assistant_id]
        try:
             utterance_ids = self.tokenizer.encode(cleaned_text, add_special_tokens=False).ids
        except Exception as e:
             logging.error(f"Error encoding stream text: {e}")
             return None # Skip on encoding error

        # Check length: BOS + Role + Utterance + EOS <= MaxLength
        if len(formatted_ids) + len(utterance_ids) + 1 > self.max_length:
            # Truncate utterance_ids if too long
            available_len = self.max_length - len(formatted_ids) - 1 # Space for EOS
            if available_len <= 0:
                 logging.debug("Skipping stream item because even BOS+Role+EOS exceeds max_length.")
                 return None # Cannot even fit BOS+Role+EOS
            utterance_ids = utterance_ids[:available_len]

        formatted_ids.extend(utterance_ids)
        formatted_ids.append(self.eos_id)

        # Ensure sequence is valid for producing labels
        if len(formatted_ids) < 2: return None

        input_ids = formatted_ids[:-1]
        labels = formatted_ids[1:]

        if not input_ids: return None

        return {"input_ids": input_ids, "labels": labels}


    def __iter__(self):
        # Combine all cached items into one list and shuffle initially
        all_cached_items = [item for sublist in self.loaded_datasets for item in sublist]
        random.shuffle(all_cached_items)
        cached_iter = iter(all_cached_items)
        num_cached = len(all_cached_items)
        processed_cached_count = 0

        # Create iterators for streamable datasets
        stream_iters = [iter(ds) for ds in self.streamable_datasets]

        # Create a combined list of all dataset sources (cached + streams)
        all_sources = list(range(len(self.loaded_datasets))) + \
                      [f"stream_{i}" for i in range(len(self.streamable_datasets))]

        # Use sampling probabilities if available, otherwise cycle equally
        use_sampling = self.sampling_probabilities is not None and len(self.sampling_probabilities) == len(all_sources)

        while True: # Loop indefinitely, training loop will break it
            source_idx = -1
            if use_sampling:
                 # Choose source based on sampling probabilities
                 source_idx = random.choices(range(len(all_sources)), weights=self.sampling_probabilities, k=1)[0]
            else:
                 # Simple round-robin cycling if no probabilities
                 source_idx = (processed_cached_count + sum(1 for _ in stream_iters)) % len(all_sources) # Poor approximation, better to use random choice

            chosen_source_id = all_sources[source_idx]

            item = None
            tokenized_item = None

            try:
                 if isinstance(chosen_source_id, int): # It's an index into self.loaded_datasets (represents combined cache)
                      if num_cached == 0: continue # Skip if no cached data
                      # Get next from the shuffled combined cache iterator
                      raw_item = next(cached_iter)
                      processed_cached_count += 1
                      tokenized_item = self._tokenize_conversation(raw_item)
                      # Reshuffle cached items when exhausted
                      if processed_cached_count >= num_cached:
                          logging.debug(f"Reshuffling {num_cached} cached items.")
                          random.shuffle(all_cached_items)
                          cached_iter = iter(all_cached_items)
                          processed_cached_count = 0

                 elif isinstance(chosen_source_id, str) and chosen_source_id.startswith("stream_"):
                      stream_idx = int(chosen_source_id.split("_")[1])
                      if stream_idx < len(stream_iters):
                          stream_iter = stream_iters[stream_idx]
                          raw_item = next(stream_iter)
                          tokenized_item = self._tokenize_stream_item(raw_item)
                      else:
                          logging.warning(f"Invalid stream index {stream_idx}")
                          continue # Skip this iteration

                 else:
                      logging.error(f"Unknown source ID type: {chosen_source_id}")
                      continue


                 if tokenized_item:
                      yield tokenized_item

            except StopIteration:
                # Handle exhaustion of an iterator
                if isinstance(chosen_source_id, int): # Cached iterator exhausted (should be handled by reshuffle logic)
                    logging.debug("Cached iterator exhausted unexpectedly mid-cycle, reshuffling.")
                    if num_cached > 0:
                        random.shuffle(all_cached_items)
                        cached_iter = iter(all_cached_items)
                        processed_cached_count = 0
                    else:
                         # No cached data, just continue to try other sources
                         pass

                elif isinstance(chosen_source_id, str) and chosen_source_id.startswith("stream_"):
                    stream_idx = int(chosen_source_id.split("_")[1])
                    logging.debug(f"Stream dataset {stream_idx} exhausted. Resetting iterator.")
                    # Reset the specific stream iterator
                    try:
                        # Need to access the original dataset object to recreate iterator
                        original_ds = self.streamable_datasets[stream_idx]
                        stream_iters[stream_idx] = iter(original_ds)
                        # Optional: Add a small sleep or break to prevent tight loops if stream is truly finite/empty
                    except IndexError:
                        logging.error(f"Cannot reset stream iterator for index {stream_idx}.")
                    except Exception as e:
                        logging.error(f"Error resetting stream iterator {stream_idx}: {e}")

                # Continue to the next iteration to try fetching again
                continue

            except Exception as e:
                 logging.error(f"Error processing item from source {chosen_source_id}: {e}", exc_info=True)
                 # Optionally log the problematic raw_item here if needed for debugging
                 # logging.error(f"Problematic raw item: {raw_item}")
                 continue # Skip this item and try the next


    @staticmethod
    def collate_fn(batch):
        # Filter out None items potentially yielded by __iter__
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Return None if the whole batch was invalid

        max_len = max(len(item["input_ids"]) for item in batch)

        # Load tokenizer once to get pad_id - ensure path matches CONFIG
        try:
            tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])
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
            labels.append(item["labels"] + [pad_id] * pad_len)
            masks.append([1] * input_len + [0] * pad_len)

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long)
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
            lr=CONFIG["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused= (self.device.type == "cuda")
        )
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
             self.pad_id = CONFIG.get("pad_token_id", 0)
             logging.warning(f"<pad> token ID not found in tokenizer, using fallback ID: {self.pad_id}")

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.base_lr = CONFIG["learning_rate"]
        self.warmup_steps = CONFIG["warmup_steps"]
        # Add max_train_steps for potential cosine decay calculation
        self.max_train_steps = CONFIG.get("max_train_steps", -1)


    def _adjust_learning_rate(self, step):
        # Linear warmup
        if self.warmup_steps > 0 and step < self.warmup_steps:
            lr = self.base_lr * (step + 1) / self.warmup_steps
        # Optional Cosine decay after warmup
        elif self.max_train_steps > 0 and step >= self.warmup_steps:
             progress = (step - self.warmup_steps) / max(1, self.max_train_steps - self.warmup_steps)
             # Ensure progress doesn't exceed 1, clamp if necessary (though unlikely with loop condition)
             progress = min(progress, 1.0)
             # Cosine decay formula: 0.5 * base_lr * (1 + cos(pi * progress))
             # Often decays to a minimum LR, e.g., 10% of base_lr
             min_lr_ratio = 0.1
             lr = self.base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

        else: # No decay, just keep base LR after warmup
            lr = self.base_lr

        # Clamp LR to avoid extremely small values if decay goes too far (optional)
        lr = max(lr, 1e-7) # Set a minimum floor for learning rate

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_step(self, batch):
        if self.use_amp:
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        autocast_context = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=self.use_amp) if self.use_amp else nullcontext()

        with autocast_context:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)
            loss = self.criterion(logits_flat.float(), labels_flat)
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
             self.scaler.step(self.optimizer)
             self.scaler.update()
         else:
             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
             self.optimizer.step()

         self.optimizer.zero_grad(set_to_none=True)
         return current_lr


class SafetyManager:
    # (No changes needed in SafetyManager implementation itself, assuming tokenizer handled correctly)
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # More conservative list
        self.bad_words = ["kill", "murder", "suicide", "hate", "abuse", "violence", "illegal", "harm", "die", "attack", "rape", "molest", "exploit", "terror"]
        self.bad_word_ids = []
        logging.info("Initializing safety manager...")
        for word in self.bad_words:
             ids = tokenizer.encode(f" {word}", add_special_tokens=False).ids
             if ids:
                self.bad_word_ids.append(ids)
                logging.debug(f"Encoded bad word '{word}' (with space) to IDs: {ids}")
             ids_no_space = tokenizer.encode(word, add_special_tokens=False).ids
             if ids_no_space and ids_no_space != ids:
                  self.bad_word_ids.append(ids_no_space)
                  logging.debug(f"Encoded bad word '{word}' (no space) to IDs: {ids_no_space}")
             if not ids and not ids_no_space: logging.warning(f"Could not encode bad word '{word}' - skipping.")

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
        if not seq or not tokens or len(tokens) < len(seq): return False
        seq_len = len(seq)
        for i in range(len(tokens) - seq_len + 1):
            if tokens[i : i + seq_len] == seq: return True
        return False

    def content_filter(self, text_ids):
        if not isinstance(text_ids, list):
            logging.warning("Content filter received non-list input.")
            return True # Default to safe
        for bad_ids in self.bad_word_ids:
            if self.contains_sequence(text_ids, bad_ids):
                detected_word = self.tokenizer.decode(bad_ids)
                logging.warning(f"Unsafe content detected: Found sequence corresponding to '{detected_word}' (IDs: {bad_ids}).")
                return False # Unsafe
        return True # Safe

    def generate_safely(self, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        self.model.eval()
        device = next(self.model.parameters()).device
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids

        if prompt_ids and prompt_ids[0] == self.bos_id: input_ids = list(prompt_ids)
        else: input_ids = [self.bos_id] + list(prompt_ids)

        if self.assistant_id is not None: input_ids.append(self.assistant_id)
        else:
            logging.error("Assistant token ID is None, cannot properly start generation.")
            return "Error: Assistant token not found."

        generated_ids = list(input_ids)
        logging.debug(f"Starting safe generation with initial IDs: {generated_ids}")

        with torch.no_grad():
            for step in range(max_new_tokens):
                current_input_ids = generated_ids[-CONFIG["max_seq_len"]:]
                current_input_tensor = torch.tensor([current_input_ids]).to(device)
                attention_mask = torch.ones_like(current_input_tensor)

                try:
                    outputs = self.model(current_input_tensor, attention_mask=attention_mask)
                    next_token_logits = outputs[:, -1, :]
                except Exception as e:
                     logging.error(f"Model forward pass failed during generation: {e}")
                     break

                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_k > 0 and top_k < next_token_logits.size(-1):
                    v, _ = torch.topk(next_token_logits, top_k)
                    safe_logits = torch.nan_to_num(next_token_logits, nan=-float('inf'), posinf=float('inf'), neginf=-float('inf'))
                    threshold = v[:, [-1]]
                    safe_logits[safe_logits < threshold] = -float('Inf')
                    next_token_logits = safe_logits

                probs = torch.softmax(next_token_logits, dim=-1)
                if torch.isnan(probs).any():
                     logging.warning("NaN detected in probabilities before sampling. Replacing with uniform distribution.")
                     probs = torch.ones_like(probs) / probs.size(-1)

                next_token_id = torch.multinomial(probs, num_samples=1).item()

                potential_sequence_ids = generated_ids + [next_token_id]
                if not self.content_filter(potential_sequence_ids):
                    logging.warning(f"Potential unsafe token ({next_token_id}, '{self.tokenizer.decode([next_token_id])}') blocked POST-sampling. Stopping generation.")
                    break

                generated_ids.append(next_token_id)

                if next_token_id == self.eos_id:
                    logging.debug(f"EOS token generated at step {step+1}. Stopping generation.")
                    break

                if step == max_new_tokens - 1:
                     logging.debug("Max new tokens reached. Stopping generation.")
                     if generated_ids[-1] != self.eos_id and self.eos_id is not None:
                         generated_ids.append(self.eos_id)

        self.model.train()
        start_index = len(input_ids)
        response_ids = generated_ids[start_index:]
        decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return decoded_text

    def debug_generation(self, prompt="<user> Tell me about your hobbies."):
         logging.info(f"\n--- Debug Generation & Safety Check ---")
         if not prompt.strip().endswith("</s>"):
              if not prompt.strip().endswith("<user>") and not prompt.strip().endswith("<assistant>"):
                   prompt = prompt.strip() + " </s>"
              else:
                   prompt = prompt.strip() + " </s>"
         if prompt.startswith("<s>"): prompt = prompt[len("<s>"):].strip()

         generated_response = self.generate_safely(prompt, max_new_tokens=60, temperature=0.7, top_k=50)
         logging.info(f"Prompt Sent: '{prompt}'")
         logging.info(f"Generated Response: '{generated_response}'")
         logging.info("\n--- End Debug Generation ---\n")


class CheckpointManager:
    def __init__(self):
        self.checkpoint_dir = CONFIG["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoint directory set to: {self.checkpoint_dir}")

    def save(self, model, optimizer, step):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "") # Use updated base name logic if needed
        step_str = str(step)
        filename = f"hrom_{prefix}_step{step_str}_{timestamp}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step if isinstance(step, int) else -1,
            "config": CONFIG
        }
        logging.info(f"Saving checkpoint to {path}...")
        try:
            torch.save(state, path)
            logging.info(f"Checkpoint saved successfully at step {step_str}.")
            self._cleanup_old_checkpoints()
        except Exception as e:
            logging.error(f"Failed to save checkpoint '{path}': {e}")

    def _cleanup_old_checkpoints(self):
        max_checkpoints = CONFIG.get("max_checkpoints", 5)
        if max_checkpoints <= 0: return

        try:
            prefix = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
            pattern = re.compile(rf"hrom_{prefix}_step(\d+|.+)_(\d{{8}}_\d{{6}})\.pt")
            checkpoints = []
            for f in os.listdir(self.checkpoint_dir):
                 match = pattern.match(f)
                 if match:
                      filepath = os.path.join(self.checkpoint_dir, f)
                      checkpoints.append((filepath, os.path.getmtime(filepath)))

            checkpoints.sort(key=lambda x: x[1])
            num_to_delete = len(checkpoints) - max_checkpoints
            if num_to_delete > 0:
                for i in range(num_to_delete):
                    file_to_remove, _ = checkpoints[i]
                    try:
                        os.remove(file_to_remove)
                    except OSError as e:
                        logging.error(f"Error removing checkpoint {file_to_remove}: {e}")
        except Exception as e:
            logging.error(f"Error during checkpoint cleanup: {e}")


    def load_latest(self, model, optimizer):
        try:
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
                return 0

            checkpoints.sort(key=lambda x: x[1], reverse=True)
            latest_checkpoint_path, _ = checkpoints[0]
            logging.info(f"Loading latest checkpoint from: {latest_checkpoint_path}")
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)

            loaded_config = checkpoint.get("config", {})
            critical_keys = ["dim", "n_layers", "n_heads", "ff_dim", "vocab_size", "max_seq_len", "tokenizer_name"]
            mismatched_keys = []
            if loaded_config:
                for key in critical_keys:
                    if key in loaded_config and key in CONFIG and loaded_config[key] != CONFIG[key]:
                        mismatched_keys.append((key, loaded_config[key], CONFIG[key]))
                    elif key in loaded_config and key not in CONFIG:
                         mismatched_keys.append((key, loaded_config[key], "Not in current CONFIG"))
                    elif key not in loaded_config and key in CONFIG:
                         mismatched_keys.append((key, "Not in loaded CONFIG", CONFIG[key]))
                if mismatched_keys:
                    logging.warning("--- CONFIG MISMATCH DETECTED ---")
                    for key, loaded_val, current_val in mismatched_keys:
                        logging.warning(f"  - {key}: Checkpoint='{loaded_val}', Current='{current_val}'")
                    logging.warning("Proceeding with loading, but results may be unexpected.")
            else:
                logging.warning("Checkpoint does not contain configuration info. Cannot check compatibility.")

            try:
                 model.load_state_dict(checkpoint['model'], strict=True)
            except RuntimeError as e:
                 logging.error(f"Failed to load model state_dict: {e}. Starting training from scratch.")
                 return 0

            try:
                 optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                 logging.warning(f"Could not load optimizer state_dict: {e}. Optimizer state will be reset.")
                 optimizer.state = defaultdict(dict) # Reset state
            except Exception as e:
                 logging.error(f"Unexpected error loading optimizer state: {e}. Starting training from scratch.")
                 return 0

            start_step = checkpoint.get('step', 0)
            start_step = max(0, start_step) + 1 if isinstance(start_step, int) else 0

            logging.info(f"Checkpoint loaded successfully. Resuming from optimizer step {start_step}.")
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        try: state[k] = v.to(map_location)
                        except Exception as e: logging.error(f"Failed to move optimizer tensor '{k}' to device '{map_location}': {e}")
            return start_step

        except FileNotFoundError:
            logging.info(f"No checkpoint directory '{self.checkpoint_dir}' or files found. Starting training from scratch.")
            return 0
        except Exception as e:
            logging.error(f"Error loading checkpoint from '{self.checkpoint_dir}': {e}. Starting training from scratch.")
            return 0


# --- Training Function (Modified for IterableDataset) ---

def train():
    # Update log message for the added dataset
    logging.info("Starting HROM training process on combined datasets (including Universal Transformers)...")
    logging.info(f"Configuration: {CONFIG}")

    # --- Tokenizer Setup ---
    tokenizer_trainer = TokenizerTrainer()
    tokenizer_path = tokenizer_trainer.tokenizer_path
    if not os.path.exists(tokenizer_path):
        logging.info(f"Combined tokenizer '{CONFIG['tokenizer_name']}' not found. Training tokenizer...")
        try:
            # Pass only non-streaming datasets for tokenizer training
            tokenizer_datasets = [ds for ds in CONFIG["datasets"] if ds != "future-technologies/Universal-Transformers-Dataset"]
            tokenizer_trainer.train(tokenizer_datasets)
        except Exception as e:
             logging.error(f"Failed during tokenizer training: {e}", exc_info=True)
             return
    else:
        logging.info(f"Loading existing combined tokenizer from {tokenizer_path}")

    try:
        tokenizer = tokenizer_trainer.get_tokenizer()
        CONFIG['pad_token_id'] = tokenizer.token_to_id("<pad>")
        CONFIG['bos_token_id'] = tokenizer.token_to_id("<s>")
        CONFIG['eos_token_id'] = tokenizer.token_to_id("</s>")
        logging.info(f"Loaded tokenizer. Vocab size: {tokenizer.get_vocab_size()}. Special IDs: PAD={CONFIG['pad_token_id']}, BOS={CONFIG['bos_token_id']}, EOS={CONFIG['eos_token_id']}")
    except (FileNotFoundError, ValueError) as e:
         logging.error(f"Failed to load tokenizer: {e}. Cannot continue.")
         return

    # --- Model Initialization ---
    logging.info("Initializing HROM model...")
    if CONFIG['vocab_size'] != tokenizer.get_vocab_size():
         logging.warning(f"Config vocab_size ({CONFIG['vocab_size']}) differs from tokenizer vocab size ({tokenizer.get_vocab_size()}). Using tokenizer's size.")
         CONFIG['vocab_size'] = tokenizer.get_vocab_size()
    model = HROM()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model initialized. Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Parameters (Millions): Total={total_params/1e6:.2f}M, Trainable={trainable_params/1e6:.2f}M")

    # --- Dataset and DataLoader ---
    logging.info("Setting up combined iterable dataset and dataloader...")
    try:
        # Pre-caching/download check isn't as relevant for streaming, but good for others
        logging.info("Checking cache for non-streaming datasets...")
        for ds_name in CONFIG["datasets"]:
             if ds_name != "future-technologies/Universal-Transformers-Dataset":
                 logging.info(f"Checking cache for '{ds_name}'...")
                 try:
                      _ = load_dataset(ds_name, split="train[:1]", download_mode="reuse_cache_if_exists", trust_remote_code=True)
                 except Exception as e:
                      logging.error(f"Could not pre-check dataset '{ds_name}': {e}")
        logging.info("Dataset download/cache check presumed complete for non-streaming.")

        dataset = CombinedChatIterableDataset(tokenizer)

        # Note: shuffle=True is ignored for IterableDataset by DataLoader. Shuffling is handled within the dataset's __iter__.
        # num_workers > 0 with IterableDataset requires careful implementation of __iter__ to be multi-process safe.
        # Start with num_workers=0 for simplicity and safety with streaming.
        num_workers = 0
        logging.info(f"Using num_workers={num_workers} for DataLoader with IterableDataset.")
        dataloader = DataLoader(
             dataset,
             batch_size=CONFIG["batch_size"],
             collate_fn=CombinedChatIterableDataset.collate_fn, # Use static method
             num_workers=num_workers,
             pin_memory=torch.cuda.is_available() and num_workers == 0, # pin_memory only works with num_workers=0 for iterable
        )
    except Exception as e:
         logging.error(f"Failed to initialize dataset/dataloader: {e}", exc_info=True)
         return

    # --- Trainer, Checkpoint, Safety ---
    logging.info("Initializing Trainer, Checkpoint Manager, and Safety Manager...")
    trainer_obj = HROMTrainer(model, tokenizer)
    checkpoint_manager = CheckpointManager()
    safety = SafetyManager(model, tokenizer)

    # --- Load Checkpoint ---
    start_optimizer_step = checkpoint_manager.load_latest(model, trainer_obj.optimizer)
    model.to(trainer_obj.device) # Ensure model is on correct device

    # --- Training Loop (Step-based) ---
    logging.info(f"Starting training loop from optimizer step {start_optimizer_step} for max {CONFIG['max_train_steps']} steps...")
    optimizer_step = start_optimizer_step
    total_loss_accum = 0.0
    model.train()
    data_iterator = iter(dataloader) # Get iterator from DataLoader

    max_steps = CONFIG["max_train_steps"]

    while optimizer_step < max_steps:
        # Gradient Accumulation
        batch_losses = []
        for accum_step in range(CONFIG["grad_accum_steps"]):
            try:
                batch = next(data_iterator)
                if batch is None: # Should be handled by collate_fn returning None for empty batches
                     logging.warning(f"Skipping None batch returned by dataloader at Opt Step {optimizer_step}, Accum Step {accum_step+1}")
                     continue
            except StopIteration:
                 logging.warning("DataLoader iterator exhausted. Resetting iterator.")
                 # This might happen if the underlying iterable dataset stops unexpectedly,
                 # though our CombinedChatIterableDataset __iter__ loops indefinitely.
                 # Resetting might be useful if workers cause issues.
                 data_iterator = iter(dataloader)
                 try:
                    batch = next(data_iterator)
                    if batch is None: continue
                 except StopIteration:
                      logging.error("DataLoader iterator exhausted immediately after reset. Stopping training.")
                      # Save before exiting
                      checkpoint_manager.save(model, trainer_obj.optimizer, f"{optimizer_step}_stopiter_err")
                      return

            # Perform train step for one batch
            loss = trainer_obj.train_step(batch)

            # Check for NaN/Inf loss
            if loss is None or torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                 logging.error(f"NaN/Inf loss ({loss}) detected at Opt Step {optimizer_step}, Accum Step {accum_step+1}. Stopping.")
                 checkpoint_manager.save(model, trainer_obj.optimizer, f"{optimizer_step}_naninf_err")
                 return # Stop training

            batch_losses.append(loss)

        # Check if any valid batches were processed in the accumulation cycle
        if not batch_losses:
            logging.warning(f"Skipping optimizer step {optimizer_step} as no valid batches were processed in accumulation cycle.")
            continue

        # --- Optimizer Step ---
        # Calculate average loss over the accumulation steps that had valid losses
        avg_loss_accum = sum(batch_losses) / len(batch_losses)
        # Clip gradients and perform optimizer step, also adjusts LR
        current_lr = trainer_obj.clip_and_step(optimizer_step)

        # Logging
        if optimizer_step % CONFIG["debug_interval"] == 0:
            # Log every debug_interval steps
            logging.info(f"Opt Step {optimizer_step}/{max_steps} | Avg Accum Loss: {avg_loss_accum:.4f} | LR: {current_lr:.2e}")
            if optimizer_step > 0 and optimizer_step % (CONFIG["debug_interval"] * 5) == 0:
                 safety.debug_generation("<user> Hi there! How are you doing today?")

        # Checkpointing
        if optimizer_step > 0 and optimizer_step % CONFIG["checkpoint_interval"] == 0:
            logging.info(f"Checkpoint interval reached at optimizer step {optimizer_step}.")
            checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)
            # Optional: Run a generation check after saving checkpoint
            safety.debug_generation("<user> Hi! How are you?")

        optimizer_step += 1 # Increment optimizer step count *after* a successful step

    # --- End of Training ---
    logging.info(f"Training finished after reaching {optimizer_step}/{max_steps} optimizer steps.")
    # Final save
    logging.info("Saving final model state...")
    checkpoint_manager.save(model, trainer_obj.optimizer, f"final_step{optimizer_step}")
    # Final debug generation
    safety.debug_generation("<user> What did you learn?")


if __name__ == "__main__":
    train()