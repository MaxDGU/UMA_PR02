#!/usr/bin/env python3
"""
Phase 3: Train Transformer on UMA Synthetic Traces

This script trains a transformer to match UMA behavior using traces from
curriculum-trained UMA students.

Input:
- uma_synthetic_traces.csv (~1M traces from trained UMA students)

Evaluation:
- Test on sp2013 (same 16 problems as paper)
- Compare transformer accuracy to verified UMA accuracy from UMA_PR02_fork/verification_full_results.csv

Usage:
    python train_transformer_phase3.py [--epochs 50] [--batch_size 128]
"""

import sys
import os
import warnings
import argparse
from datetime import datetime
import gc
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from fractions import Fraction
from typing import List, Tuple

# Plotting style
sns.set_style('ticks')
sns.set_context('paper')
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['figure.dpi'] = 100
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UMA_MODEL_DIR = os.path.join(SCRIPT_DIR, 'UMA_PR02', '1. Model')
# Will use uma_traces_all.csv from the merged parallel generation
SYNTHETIC_TRACES_PATH = os.path.join(SCRIPT_DIR, 'uma_traces_all.csv')
# Verified UMA results (from trained models, not the old buggy file)
VERIFIED_UMA_PATH = os.path.join(SCRIPT_DIR, 'UMA_PR02_fork', 'verification_full_results.csv')
TEST_SETS_DIR = os.path.join(SCRIPT_DIR, 'UMA_PR02_fork', '1. Model', 'Problem Sets Testing')
SP2013_PATH = os.path.join(UMA_MODEL_DIR, 'Problem Sets Testing', 'sp2013.csv')
BSS2021_PATH = os.path.join(TEST_SETS_DIR, 'bss2021.csv')
SD_ADD_PATH = os.path.join(TEST_SETS_DIR, 'sd_add.csv')
SD_MUL_PATH = os.path.join(TEST_SETS_DIR, 'sd_mul.csv')

TEST_SET_PATHS = {
    'sp2013': SP2013_PATH,
    'bss2021': BSS2021_PATH,
    'sd_add': SD_ADD_PATH,
    'sd_mul': SD_MUL_PATH,
}

# Output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'phase3_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# Tokenizer (same as bbt_phase1_gomath.py)
# ==============================================================================

class FractionTokenizer:
    def __init__(self):
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '<mask>',
            '<problem>', '</problem>', '<step>', '</step>',
            '<strategy>', '</strategy>', '<goal>', '</goal>',
            '<rule>', '</rule>', '<effect>', '</effect>',
            '<answer>', '</answer>',
            '<student>', '</student>',
            '<rules>', '</rules>',
            '<goals>', '</goals>',
            '<exec>', '</exec>',
            '<work>', '</work>']
        # Extended range to handle larger intermediate results (e.g., 42/15 → 42)
        self.digits = [str(i) for i in range(1000)]
        # Added '.' for decimal handling, '=' and '|' for scratchpad work steps
        self.operators = ['/', '+', '-', '*', ':', '?', '→', '.', '=', '|']
        self.student_params = [
            # Coarse bins (backward compatible)
            'g_low', 'g_mid', 'g_high',
            'd_low', 'd_mid', 'd_high',
            'rt_3', 'rt_4', 'rt_5', 'rt_6',
            'ice_0', 'ice_25', 'ice_50', 'ice_75', 'ice_100',
            # Fine bins (match exact UMA parameter grid)
            'g_01', 'g_02', 'g_03', 'g_04', 'g_05',
            'g_06', 'g_07', 'g_08', 'g_09', 'g_10',
            'd_1', 'd_3', 'd_5', 'd_7', 'd_9',
        ]
        self.strategy_names = [
            'KDON_AS', 'KDON_OG', 'CDON_AS', 'CDON_OG',
            'ONOD_M', 'ONOD_OG', 'CROP_M', 'ICDM_D', 'ICDM_OG',
            'ADBD_AS', 'ADBD_OG', 'ARAD_M', 'ARAD_OG',
            'H2V_WN',
            'OTHER',
        ]
        self.rule_names = [
            'add_fact', 'mul_fact', 'deferred_action', 'finish_problem', 'op1_none_to_zero',
            'op2_none_to_zero', 'larger_first_add_mul', 'add_to_zero', 'sub_zero_from',
            'mul_by_zero', 'mul_by_one', 'acc_once', 'acc_skip', 'acc_extra', 'acc_end',
            'acc_count', 'acc_add', 'sub_fact', 'sub_LbS', 'div_calculator', 'div_LbS',
            'div_LbS_drop_rem', 'div_drop_rem', 'list_aggregation_start', 'list_aggregation_two',
            'list_aggregation_set', 'list_aggregation_calc', 'list_aggregation_skip',
            'list_aggregation_zero', 'list_aggregation_end', 'H2V_WN', 'align_right', 'VA_start',
            'choose_VAS_AS', 'VA_next_calc', 'VA_lone_carry', 'VS_next_calc', 'VS_borrow_start',
            'VS_borrow_to', 'VS_borrow_from_nonzero', 'VS_borrow_from_zero', 'VS_next_calc_swap',
            'VS_borrow_from_fail', 'VAS_finish', 'VAS_end_calc', 'VAS_do_carry', 'VAS_add_carry',
            'VAS_shift_attn', 'choose_VM_M', 'VM_next_calc', 'VM_lone_carry', 'VM_finish_sd',
            'VM_end_calc', 'VM_do_carry', 'VM_add_carry', 'VM_shift1', 'VM_shift2', 'VM_new_row',
            'VM_add_parts', 'VM_shift2_no_zeros', 'ADBD_AS', 'ADBD_OG', 'ARAD_M', 'ARAD_OG',
            'align_dec_no_zeros', 'align_dec_app_zeros', 'bring_dec', 'add_dd', 'add_dd_skip',
            'P2F', 'N2F_fra', 'N2F_wn', 'N2F_mix', 'KDON_AS', 'CDON_AS', 'ONOD_M', 'ICDM_D',
            'operate_nums', 'operate_dens', 'pass_den', 'convert_CD', 'convert_CD_LCM', 'get_LCM',
            'convert_fra_to_den', 'invert_op2', 'div_to_mul', 'KDON_OG', 'CDON_OG', 'ONOD_OG',
            'ICDM_OG', 'CROP_M', 'convert_CD_omit_nums', 'invert_rand', 'invert_fail',
            'div_to_mul_denied', 'check_simplify', 'skip_simplify', 'get_GCD', 'simplify_fraction',
            'cannot_simplify', 'no_exec'
        ]
        self.ws_keys = ['problem:', 'op1_num:', 'op1_den:', 'op2_num:', 'op2_den:',
                        'ans_num:', 'ans_den:', 'answer:']
        self.vocab = (self.special_tokens + self.digits + self.operators +
                      self.student_params + self.strategy_names +
                      self.rule_names + self.ws_keys + ['None'])
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}
        self.pad_token_id = self.token_to_id['<pad>']
        self.unk_token_id = self.token_to_id['<unk>']
        self.bos_token_id = self.token_to_id['<bos>']
        self.eos_token_id = self.token_to_id['<eos>']

    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        ids = [self.bos_token_id]
        for tok in tokens:
            if tok in self.token_to_id:
                ids.append(self.token_to_id[tok])
            else:
                for char in tok:
                    ids.append(self.token_to_id.get(char, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        return ' '.join(self.id_to_token.get(i, '<unk>') for i in ids
                       if self.id_to_token.get(i) not in ['<bos>', '<eos>', '<pad>'])

    def __len__(self):
        return len(self.vocab)


# ==============================================================================
# Dataset
# ==============================================================================

def bin_g(g: float, fine: bool = False) -> str:
    if fine:
        # Map to exact grid value: 0.01 -> g_01, ..., 0.10 -> g_10
        idx = int(round(g * 100))
        idx = max(1, min(10, idx))
        return f'g_{idx:02d}'
    if g < 0.04:
        return 'g_low'
    elif g < 0.08:
        return 'g_mid'
    else:
        return 'g_high'


def bin_d(d: float, fine: bool = False) -> str:
    if fine:
        # Map to exact grid value: 0.1 -> d_1, 0.3 -> d_3, ..., 0.9 -> d_9
        idx = int(round(d * 10))
        idx = max(1, min(9, idx))
        return f'd_{idx}'
    if d <= 0.35:
        return 'd_low'
    elif d <= 0.65:
        return 'd_mid'
    else:
        return 'd_high'


def bin_ice(ice: float) -> str:
    if ice <= 12.5:
        return 'ice_0'
    elif ice <= 37.5:
        return 'ice_25'
    elif ice <= 62.5:
        return 'ice_50'
    elif ice <= 87.5:
        return 'ice_75'
    else:
        return 'ice_100'


def bin_rt_mu(rt_mu: float) -> str:
    if rt_mu <= 3.5:
        return 'rt_3'
    elif rt_mu <= 4.5:
        return 'rt_4'
    elif rt_mu <= 5.5:
        return 'rt_5'
    else:
        return 'rt_6'


class SyntheticTracesDataset(Dataset):
    """
    Dataset for synthetic UMA traces with decoder-only (GPT-style) format.

    Full sequence: <bos> <student> params </student> <problem> prob </problem> <strategy> ... </strategy> <goals> ... </goals> <exec> ... </exec> <answer> ... </answer> <eos>

    We compute loss only on tokens after </problem> (the target portion).
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 100, scratchpad: bool = False, fine_bins: bool = False):
        self.samples = []
        self.tokenizer = tokenizer
        self.scratchpad = scratchpad

        for _, row in df.iterrows():
            prob = str(row['prob']).strip()
            resp = str(row.get('answer', row.get('resp', '?'))).strip()

            if resp == '?' or resp == 'nan' or not resp:
                continue

            g_bin = bin_g(row['g'], fine=fine_bins)
            d_bin = bin_d(row['d'], fine=fine_bins)
            rt_bin = bin_rt_mu(row['rt_mu']) if pd.notna(row.get('rt_mu')) else 'rt_4'
            ice_bin = bin_ice(row['ice'])
            strategy = str(row['strategy'])
            goals_str = str(row['goals']) if pd.notna(row.get('goals')) and row['goals'] else ''
            exec_str = str(row['exec']) if pd.notna(row.get('exec')) and row['exec'] else ''

            # Context (prompt): student params + problem
            context_text = f"<student> {g_bin} {d_bin} {rt_bin} {ice_bin} </student> <problem> {prob} </problem>"

            # Replace nan exec with no_exec in all modes
            if not exec_str or exec_str == 'nan':
                exec_str = 'no_exec'

            if scratchpad:
                # Scratchpad mode: goals + exec + work (all three)
                work_str = str(row['work']) if pd.notna(row.get('work')) and row.get('work') else ''
                if work_str and work_str != 'nan':
                    target_text = (f"<strategy> {strategy} </strategy> "
                                   f"<goals> {goals_str} </goals> "
                                   f"<exec> {exec_str} </exec> "
                                   f"<work> {work_str} </work> "
                                   f"<answer> {resp} </answer>")
                else:
                    # Fallback: goals + exec only if no work column
                    target_text = (f"<strategy> {strategy} </strategy> "
                                   f"<goals> {goals_str} </goals> "
                                   f"<exec> {exec_str} </exec> "
                                   f"<answer> {resp} </answer>")
            else:
                # Standard mode: goals + exec only
                target_text = (f"<strategy> {strategy} </strategy> "
                               f"<goals> {goals_str} </goals> "
                               f"<exec> {exec_str} </exec> "
                               f"<answer> {resp} </answer>")

            # Full sequence for decoder-only
            full_text = context_text + " " + target_text
            full_ids = tokenizer.encode(full_text)  # includes BOS and EOS

            # Find where target starts (after </problem>)
            context_ids = tokenizer.encode(context_text)
            context_len = len(context_ids) - 1  # exclude EOS from context

            if len(full_ids) <= max_len:
                self.samples.append((full_ids, context_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_ids, context_len = self.samples[idx]
        return (torch.tensor(full_ids[:-1], dtype=torch.long),   # input (no EOS)
                torch.tensor(full_ids[1:], dtype=torch.long),    # target (no BOS)
                context_len)  # where to start computing loss


def augment_commutative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment training data by swapping operands for commutative operations (+ and *).

    For addition a/b + c/d, the augmented version is c/d + a/b with same trace.
    For multiplication a/b * c/d, the augmented version is c/d * a/b with same trace.
    """
    import re

    augmented_rows = []
    commutative_ops = ['+', '*']

    for _, row in df.iterrows():
        prob = str(row['prob'])
        op = row['operation']

        if op in commutative_ops:
            # Parse the problem
            if op == '+':
                parts = prob.split('+')
            else:  # op == '*'
                parts = prob.split('*')

            if len(parts) == 2:
                frac1, frac2 = parts[0].strip(), parts[1].strip()
                # Create swapped problem
                swapped_prob = f"{frac2}{op}{frac1}"

                # Create augmented row
                aug_row = row.copy()
                aug_row['prob'] = swapped_prob
                augmented_rows.append(aug_row)

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        combined = pd.concat([df, aug_df], ignore_index=True)
        print(f"  Augmentation: added {len(augmented_rows)} swapped examples")
        print(f"  New total: {len(combined)} traces")
        return combined

    return df


def apply_curriculum_ordering(df: pd.DataFrame, curriculum: str) -> pd.DataFrame:
    """
    Apply curriculum ordering to training data.

    Args:
        df: DataFrame with training traces
        curriculum: Ordering strategy
            - 'random': Random shuffle (default, no curriculum)
            - 'blocked_complexity': Order by sequence complexity (goals+exec length)
            - 'blocked_ed_first': ED problems first, then UD
            - 'blocked_student': Order by student ability (high g, high d first)

    Returns:
        Reordered DataFrame
    """
    import re

    if curriculum == 'random':
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    elif curriculum == 'blocked_complexity':
        # Compute sequence complexity (goals + exec token count)
        def get_complexity(row):
            goals_len = len(str(row['goals']).split()) if pd.notna(row.get('goals')) else 0
            exec_len = len(str(row['exec']).split()) if pd.notna(row.get('exec')) else 0
            return goals_len + exec_len

        df = df.copy()
        df['_complexity'] = df.apply(get_complexity, axis=1)
        df = df.sort_values('_complexity').reset_index(drop=True)
        df = df.drop(columns=['_complexity'])
        print(f"  Curriculum: blocked_complexity (short sequences first)")
        return df

    elif curriculum == 'blocked_ed_first':
        # Classify ED vs UD
        def get_denom_type(prob):
            fractions = re.findall(r'(\d+)/(\d+)', str(prob))
            if len(fractions) >= 2:
                d1, d2 = int(fractions[0][1]), int(fractions[1][1])
                return 0 if d1 == d2 else 1  # ED=0 (first), UD=1 (second)
            return 1

        df = df.copy()
        df['_denom_order'] = df['prob'].apply(get_denom_type)
        # Within each block, shuffle randomly
        ed_df = df[df['_denom_order'] == 0].sample(frac=1, random_state=42)
        ud_df = df[df['_denom_order'] == 1].sample(frac=1, random_state=42)
        df = pd.concat([ed_df, ud_df], ignore_index=True)
        df = df.drop(columns=['_denom_order'])
        print(f"  Curriculum: blocked_ed_first (ED problems first, then UD)")
        print(f"    ED block: {len(ed_df)} traces, UD block: {len(ud_df)} traces")
        return df

    elif curriculum == 'blocked_student':
        # Order by student ability: weakest to strongest
        # Low g = more noise/errors, Low d = learns slower
        # Start with low-performing students, progress to high-performing
        df = df.copy()
        # Sort by g ascending (low g first = more errors), then d ascending (low d = slower learning)
        df = df.sort_values(['g', 'd'], ascending=[True, True]).reset_index(drop=True)
        print(f"  Curriculum: blocked_student (weakest to strongest)")
        print(f"    g range: {df['g'].iloc[0]:.3f} -> {df['g'].iloc[-1]:.3f}")
        print(f"    d range: {df['d'].iloc[0]:.3f} -> {df['d'].iloc[-1]:.3f}")
        return df

    else:
        print(f"  Unknown curriculum '{curriculum}', using random")
        return df.sample(frac=1, random_state=42).reset_index(drop=True)


def make_collate_fn(tokenizer=None, segment_weights=None):
    """
    Create a collate function for decoder-only model.

    Args:
        tokenizer: FractionTokenizer instance (needed if segment_weights is set)
        segment_weights: dict mapping segment names to loss weights, e.g.
            {'strategy': 5.0, 'goals': 3.0, 'exec': 3.0, 'answer': 0.5, 'structural': 1.0}
            If None, uses binary mask (0 for context, 1 for target).
    """
    # Pre-compute segment delimiter token IDs for efficiency
    seg_delimiters = None
    if segment_weights is not None and tokenizer is not None:
        seg_delimiters = {
            'strategy': (tokenizer.token_to_id['<strategy>'], tokenizer.token_to_id['</strategy>']),
            'goals': (tokenizer.token_to_id['<goals>'], tokenizer.token_to_id['</goals>']),
            'exec': (tokenizer.token_to_id['<exec>'], tokenizer.token_to_id['</exec>']),
            'answer': (tokenizer.token_to_id['<answer>'], tokenizer.token_to_id['</answer>']),
            'work': (tokenizer.token_to_id['<work>'], tokenizer.token_to_id['</work>']),
        }

    def collate_fn(batch):
        inputs, targets, context_lens = zip(*batch)
        max_len = max(len(x) for x in inputs)

        padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
        padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
        loss_mask = torch.zeros(len(inputs), max_len, dtype=torch.float)

        for i, (inp, tgt, ctx_len) in enumerate(zip(inputs, targets, context_lens)):
            padded_inputs[i, :len(inp)] = inp
            padded_targets[i, :len(tgt)] = tgt

            if segment_weights is not None and seg_delimiters is not None:
                # Assign per-token weights based on which segment the target token is in.
                # We scan the INPUT sequence (since target is shifted by 1, the segment
                # delimiters in input tell us what segment the corresponding target token
                # belongs to).
                structural_w = segment_weights.get('structural', 1.0)
                inp_list = inp.tolist()
                seq_len = len(inp_list)

                # Find segment boundaries in input sequence
                # A token at position j in target corresponds to predicting inp[j+1],
                # but target[j] = full_ids[j+1]. The segment of target[j] is determined
                # by where j falls relative to delimiters in the input.
                current_segment = 'structural'
                for j in range(ctx_len - 1, len(tgt)):
                    tok_id = inp_list[j] if j < seq_len else 0
                    # Check if this token is a segment opener/closer
                    for seg_name, (open_id, close_id) in seg_delimiters.items():
                        if tok_id == open_id:
                            current_segment = seg_name
                            break
                        elif tok_id == close_id:
                            current_segment = 'structural'
                            break
                    loss_mask[i, j] = segment_weights.get(current_segment, structural_w)
            else:
                # Binary mask: 0 for context, 1 for target
                loss_mask[i, ctx_len-1:len(tgt)] = 1.0

        return padded_inputs, padded_targets, loss_mask

    return collate_fn


# ==============================================================================
# Model (MLC-style Encoder-Decoder)
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from MLC."""
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000.) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # token_embedding: (batch, seq_len, emb_size)
        seq_len = token_embedding.size(1)
        return self.dropout(token_embedding + self.pos_embedding[:seq_len, :])


class LegacyFractionGPT(nn.Module):
    """Old architecture (pre-3d591f3) for loading legacy checkpoints."""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.register_buffer('causal_mask', torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.dropout(self.token_embedding(x) + self.position_embedding(positions))
        mask = self.causal_mask[:T, :T]
        x = self.transformer(x, x, tgt_mask=mask, memory_mask=mask)
        return self.fc_out(x)

    def generate(self, prompt_ids, tokenizer, max_new_tokens=50, temperature=1.0):
        self.eval()
        generated = prompt_ids.clone()
        eos_id = tokenizer.token_to_id.get('</answer>', -1)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.shape[1] >= self.max_seq_len:
                    break
                logits = self.forward(generated)
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == eos_id:
                    break
        return generated


class FractionGPT(nn.Module):
    """
    Decoder-only transformer for fraction reasoning (GPT-style).

    Input: concatenated sequence of student params + problem + strategy + goals + exec + answer
    Uses causal masking, predicts next token autoregressively.
    """
    def __init__(self, vocab_size: int, hidden_size: int = 256,
                 nlayers: int = 5, nhead: int = 8, dropout_p: float = 0.1,
                 ff_mult: int = 4, activation: str = 'gelu', pad_idx: int = 0,
                 legacy_decoder: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.nlayers = nlayers
        self.legacy_decoder = legacy_decoder

        if legacy_decoder:
            # Intermediate architecture: TransformerDecoder with current naming
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=nhead,
                dim_feedforward=hidden_size * ff_mult,
                dropout=dropout_p, batch_first=True, activation=activation
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        else:
            # Current architecture: TransformerEncoder + causal mask
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=nhead,
                dim_feedforward=hidden_size * ff_mult,
                dropout=dropout_p, batch_first=True, activation=activation
            )
            self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.positional_encoding = PositionalEncoding(hidden_size, dropout_p)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        """
        Args:
            x: input sequence (batch, seq_len) - full concatenated sequence
        Returns:
            output: (batch, seq_len, vocab_size)
        """
        device = x.device
        seq_len = x.size(1)

        # Embed and add positional encoding
        x_embed = self.embedding(x)
        x_embed = self.positional_encoding(x_embed)

        # Create causal mask and padding mask
        causal_mask = self.generate_causal_mask(seq_len, device)
        padding_mask = (x == self.pad_idx)

        if self.legacy_decoder:
            out = self.decoder(
                x_embed, x_embed,
                tgt_mask=causal_mask,
                memory_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask
            )
        else:
            out = self.decoder(
                x_embed,
                mask=causal_mask,
                src_key_padding_mask=padding_mask
            )
        return self.out(out)

    def generate(self, prompt_ids, tokenizer, max_new_tokens=50, temperature=1.0):
        """Autoregressive generation given prompt."""
        self.eval()
        device = prompt_ids.device

        eos_id = tokenizer.token_to_id.get('</answer>', tokenizer.eos_token_id)
        generated = prompt_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated)
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == eos_id:
                    break

        return generated


# ==============================================================================
# Loss
# ==============================================================================

class DecoderLoss(nn.Module):
    """
    Cross-entropy loss for decoder-only model with loss masking.

    Only computes loss on target tokens (after context/prompt).
    """

    def __init__(self, tokenizer, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
            loss_mask: (batch, seq_len) - 1 for tokens to include in loss, 0 otherwise
        """
        B, T, V = logits.shape

        # Compute per-token loss
        loss_per_token = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=self.pad_idx,
            reduction='none'
        ).view(B, T)

        if loss_mask is not None:
            # Apply loss mask
            masked_loss = loss_per_token * loss_mask
            loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss


# ==============================================================================
# Training
# ==============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    n_batches = 0
    use_amp = scaler is not None and device.type == 'cuda'

    for inputs, targets, loss_mask in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                logits = model(inputs)
                loss = criterion(logits, targets, loss_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, targets, loss_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for inputs, targets, loss_mask in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    logits = model(inputs)
                    loss = criterion(logits, targets, loss_mask)
            else:
                logits = model(inputs)
                loss = criterion(logits, targets, loss_mask)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ==============================================================================
# SP2013 Evaluation - 16 question test set from UMA paper
# ==============================================================================

def compute_correct(prob):
    """Compute correct answer preserving the original number format.

    - Fraction problems (contain '/') → fraction answer (e.g., '7/15')
    - Decimal problems (contain '.') → decimal answer (e.g., '17.9')
    - Whole number problems → integer answer (e.g., '15')
    """
    try:
        prob_str = str(prob).replace(':', '/')
        if ':' in str(prob):
            parts = str(prob).split(':')
            val = eval(parts[0]) / eval(parts[1])
        else:
            val = eval(prob_str)

        # Detect number type from problem format
        if '/' in str(prob):
            # Fraction: return simplified fraction
            frac = Fraction(val).limit_denominator(1000)
            if frac.denominator == 1:
                return str(frac.numerator)
            return f"{frac.numerator}/{frac.denominator}"
        elif '.' in str(prob):
            # Decimal: return decimal answer
            result = float(val)
            formatted = f"{result:.10f}".rstrip('0').rstrip('.')
            return formatted
        else:
            # Whole number: return integer
            result = float(val)
            if result == int(result):
                return str(int(result))
            return str(result)
    except Exception:
        return '?'


def answers_match(pred_ans, correct_ans):
    """Check if predicted answer matches correct answer numerically."""
    if pred_ans == '?' or correct_ans == '?':
        return False
    try:
        if '/' in pred_ans:
            parts = pred_ans.split('/')
            pred_val = float(parts[0]) / float(parts[1])
        else:
            pred_val = float(pred_ans)

        if '/' in correct_ans:
            parts = correct_ans.split('/')
            correct_val = float(parts[0]) / float(parts[1])
        else:
            correct_val = float(correct_ans)

        return abs(pred_val - correct_val) < 1e-6
    except Exception:
        return False


def extract_region(output, open_tag, close_tag):
    if open_tag in output and close_tag in output:
        return output[output.find(open_tag)+len(open_tag):output.find(close_tag)].strip()
    return ''


def evaluate_on_sp2013(model, tokenizer, device, uma_verified_df=None, temperature=0.7, fine_bins=False, full_eval=False):
    """Evaluate transformer on sp2013 problems and compare to verified UMA."""
    model.eval()

    # Load sp2013 problems
    sp2013_df = pd.read_csv(SP2013_PATH)
    sp2013_probs = sp2013_df['prob'].tolist()

    # Get UMA accuracy from verified results
    # Map symbols to operation names used in verified file
    op_symbol_to_name = {'+': 'add', '-': 'sub', '*': 'mul', ':': 'div'}
    uma_accuracy = {}
    if uma_verified_df is not None:
        for op_symbol, op_name in op_symbol_to_name.items():
            op_df = uma_verified_df[uma_verified_df['operation'] == op_name]
            if len(op_df) > 0:
                uma_accuracy[op_symbol] = op_df['acc'].mean()

    # Student configs depend on binning mode
    if fine_bins:
        g_bins = [f'g_{i:02d}' for i in range(1, 11)]
        d_bins = [f'd_{i}' for i in [1, 3, 5, 7, 9]]
    else:
        g_bins = ['g_low', 'g_mid', 'g_high']
        d_bins = ['d_low', 'd_mid', 'd_high']
    rt_bins = ['rt_3', 'rt_4', 'rt_5', 'rt_6']
    ice_bins = ['ice_0', 'ice_25', 'ice_50', 'ice_75', 'ice_100']

    if full_eval:
        # All combinations: 10g × 5d × 4rt × 5ice = 1000 (or 3×3×4×5=180 coarse)
        student_configs = [
            (g, d, rt, ice)
            for g in g_bins for d in d_bins for rt in rt_bins for ice in ice_bins
        ]
    else:
        # Representative subset
        student_configs = [
            (g, d, rt, ice)
            for g in g_bins for d in d_bins
            for rt in ['rt_3', 'rt_6'] for ice in ['ice_0', 'ice_50', 'ice_100']
        ]
    print(f"Testing {len(student_configs)} student configurations...")

    results = []
    strategy_names = ['KDON_AS', 'KDON_OG', 'CDON_AS', 'CDON_OG',
                      'ONOD_M', 'ONOD_OG', 'CROP_M', 'ICDM_D', 'ICDM_OG']

    for prob in sp2013_probs:
        correct = compute_correct(prob)
        op = '+' if '+' in prob else '-' if '-' in prob else '*' if '*' in prob else ':'

        for g_bin, d_bin, rt_bin, ice_bin in student_configs:
            # Decoder-only: prompt is student params + problem
            prompt_text = f"<student> {g_bin} {d_bin} {rt_bin} {ice_bin} </student> <problem> {prob} </problem>"
            prompt_ids = tokenizer.encode(prompt_text)[:-1]  # remove EOS, keep BOS
            prompt_ids = torch.tensor([prompt_ids], device=device)

            with torch.no_grad():
                generated = model.generate(prompt_ids, tokenizer, max_new_tokens=60, temperature=temperature)
                output = tokenizer.decode(generated[0].tolist())

            # Extract predictions
            pred_strategy = 'OTHER'
            for s in strategy_names:
                if s in output:
                    pred_strategy = s
                    break

            pred_answer = extract_region(output, '<answer>', '</answer>').replace(' ', '')
            pred_work = extract_region(output, '<work>', '</work>')
            is_correct = answers_match(pred_answer, correct)

            results.append({
                'prob': prob,
                'operation': op,
                'student_config': f"{g_bin}_{d_bin}_{rt_bin}_{ice_bin}",
                'strategy': pred_strategy,
                'work': pred_work,
                'pred_answer': pred_answer,
                'correct_answer': correct,
                'is_correct': is_correct,
            })

    results_df = pd.DataFrame(results)

    # Compute accuracy by operation
    print(f"\n{'='*60}")
    print("SP2013 EVALUATION RESULTS")
    print(f"{'='*60}")

    op_names = {'+': 'Addition', '-': 'Subtraction', '*': 'Multiplication', ':': 'Division'}
    print(f"\n{'Operation':<15} {'Transformer':<12} {'UMA (paper)':<12} {'Match?'}")
    print("-" * 50)

    for op in ['+', '-', '*', ':']:
        op_df = results_df[results_df['operation'] == op]
        if len(op_df) > 0:
            trans_acc = op_df['is_correct'].mean()
            uma_acc = uma_accuracy.get(op, 0)
            match = 'Y' if abs(trans_acc - uma_acc) < 0.1 else 'N'
            print(f"{op_names[op]:<15} {trans_acc:>10.1%}   {uma_acc:>10.1%}   {match:>6}")

    overall_trans = results_df['is_correct'].mean()
    overall_uma = np.mean(list(uma_accuracy.values())) if uma_accuracy else 0
    print("-" * 50)
    print(f"{'Overall':<15} {overall_trans:>10.1%}   {overall_uma:>10.1%}")

    return results_df


def evaluate_on_test_set(model, tokenizer, device, test_set_name, temperature=0.7, fine_bins=False, full_eval=False):
    """Evaluate transformer on any test set (sp2013, bss2021, sd_add, sd_mul)."""
    model.eval()

    test_path = TEST_SET_PATHS.get(test_set_name)
    if test_path is None or not os.path.exists(test_path):
        print(f"  WARNING: Test set '{test_set_name}' not found at {test_path}")
        return pd.DataFrame()

    test_df = pd.read_csv(test_path)
    test_probs = test_df['prob'].tolist()
    print(f"\nEvaluating on {test_set_name}: {len(test_probs)} problems (temperature={temperature})")

    # Student configs depend on binning mode
    if fine_bins:
        g_bins = [f'g_{i:02d}' for i in range(1, 11)]
        d_bins = [f'd_{i}' for i in [1, 3, 5, 7, 9]]
    else:
        g_bins = ['g_low', 'g_mid', 'g_high']
        d_bins = ['d_low', 'd_mid', 'd_high']
    rt_bins = ['rt_3', 'rt_4', 'rt_5', 'rt_6']
    ice_bins = ['ice_0', 'ice_25', 'ice_50', 'ice_75', 'ice_100']

    if full_eval:
        student_configs = [
            (g, d, rt, ice)
            for g in g_bins for d in d_bins for rt in rt_bins for ice in ice_bins
        ]
    else:
        student_configs = [
            (g, d, rt, ice)
            for g in g_bins for d in d_bins
            for rt in ['rt_3', 'rt_6'] for ice in ['ice_0', 'ice_50', 'ice_100']
        ]

    strategy_names = [
        'KDON_AS', 'KDON_OG', 'CDON_AS', 'CDON_OG',
        'ONOD_M', 'ONOD_OG', 'CROP_M', 'ICDM_D', 'ICDM_OG',
        'ADBD_AS', 'ADBD_OG', 'ARAD_M', 'ARAD_OG', 'H2V_WN',
    ]

    results = []
    for prob in test_probs:
        correct = compute_correct(prob)
        op = '+' if '+' in str(prob) else '-' if '-' in str(prob) else '*' if '*' in str(prob) else ':'

        for g_bin, d_bin, rt_bin, ice_bin in student_configs:
            prompt_text = f"<student> {g_bin} {d_bin} {rt_bin} {ice_bin} </student> <problem> {prob} </problem>"
            prompt_ids = tokenizer.encode(prompt_text)[:-1]
            prompt_ids = torch.tensor([prompt_ids], device=device)

            with torch.no_grad():
                generated = model.generate(prompt_ids, tokenizer, max_new_tokens=60, temperature=temperature)
                output = tokenizer.decode(generated[0].tolist())

            pred_strategy = 'OTHER'
            for s in strategy_names:
                if s in output:
                    pred_strategy = s
                    break

            pred_answer = extract_region(output, '<answer>', '</answer>').replace(' ', '')
            pred_work = extract_region(output, '<work>', '</work>')
            is_correct = answers_match(pred_answer, correct)

            results.append({
                'prob': prob,
                'operation': op,
                'student_config': f"{g_bin}_{d_bin}_{rt_bin}_{ice_bin}",
                'strategy': pred_strategy,
                'work': pred_work,
                'pred_answer': pred_answer,
                'correct_answer': correct,
                'is_correct': is_correct,
                'test_set': test_set_name,
            })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        print(f"\n{'='*60}")
        print(f"{test_set_name.upper()} EVALUATION RESULTS")
        print(f"{'='*60}")

        op_names = {'+': 'Addition', '-': 'Subtraction', '*': 'Multiplication', ':': 'Division'}
        ops_present = sorted(results_df['operation'].unique())
        for op in ops_present:
            op_df = results_df[results_df['operation'] == op]
            if len(op_df) > 0:
                acc = op_df['is_correct'].mean()
                print(f"  {op_names.get(op, op):<15} {acc:>8.1%}")

        overall = results_df['is_correct'].mean()
        print(f"  {'Overall':<15} {overall:>8.1%}")

    return results_df


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 3: Train on Synthetic Traces')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (MLC default: 50)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (MLC default: 1e-3)')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension (MLC default: 128)')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=5, help='Number of decoder layers (decoder-only)')
    parser.add_argument('--max_samples', type=int, default=None, help='Max training samples (for debugging)')
    parser.add_argument('--curriculum', type=str, default='random',
                        choices=['random', 'blocked_complexity', 'blocked_ed_first', 'blocked_student'],
                        help='Curriculum ordering strategy (default: random)')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for output files (e.g., "large")')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs for learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation (operand swapping for +/*)')
    parser.add_argument('--extra_data', type=str, default=None, help='Path to additional training data CSV')
    parser.add_argument('--weighted_loss', action='store_true', help='Upweight reasoning tokens (strategy=5, goals=3, exec=3, answer=0.5)')
    parser.add_argument('--mask_answer', action='store_true', help='Zero out answer tokens in loss (train only on reasoning)')
    parser.add_argument('--scratchpad', action='store_true', help='Use scratchpad traces (<work> section) instead of goals/exec')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to pretrained model checkpoint to load')
    parser.add_argument('--eval_temperature', type=float, default=0.7, help='Temperature for SP2013 generation (default: 0.7)')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only evaluate loaded checkpoint on SP2013')
    parser.add_argument('--data_path', type=str, default=None, help='Path to training data CSV (default: uma_traces_all.csv)')
    parser.add_argument('--fine_bins', action='store_true',
                        help='Use fine-grained student parameter bins (10 g × 5 d) instead of coarse (3 g × 3 d)')
    parser.add_argument('--full_eval', action='store_true',
                        help='Evaluate all 1000 student configs (10g × 5d × 4rt × 5ice) instead of representative subset')
    parser.add_argument('--test_sets', type=str, nargs='+', default=['sp2013'],
                        choices=['sp2013', 'bss2021', 'sd_add', 'sd_mul'],
                        help='Test sets to evaluate on (default: sp2013)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"{'='*70}")
    print("PHASE 3: Train Transformer on UMA Synthetic Traces")
    print(f"Curriculum: {args.curriculum}")
    if args.scratchpad:
        print("Mode: SCRATCHPAD (using <work> section with intermediate arithmetic)")
    if args.fine_bins:
        print("Student bins: FINE (10 g × 5 d = 50 types, matching UMA parameter grid)")
    print(f"{'='*70}")

    # Early eval-only path: skip all data loading, just load model and evaluate
    if args.eval_only:
        if not args.load_checkpoint:
            print("ERROR: --eval_only requires --load_checkpoint")
            return

        tokenizer = FractionTokenizer()
        model = FractionGPT(
            vocab_size=len(tokenizer),
            hidden_size=args.d_model,
            nlayers=args.n_layers,
            nhead=args.n_heads,
            dropout_p=args.dropout,
            pad_idx=tokenizer.pad_token_id
        ).to(device)

        ckpt_path = args.load_checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(SCRIPT_DIR, ckpt_path)
        print(f"\nLoading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        # Handle vocab size mismatch (older checkpoints may have fewer tokens)
        model_vocab = model.embedding.weight.shape[0]
        ckpt_vocab = cleaned['embedding.weight'].shape[0]
        if ckpt_vocab != model_vocab:
            print(f"  Vocab size mismatch: checkpoint={ckpt_vocab}, model={model_vocab}. Padding new tokens.")
            for key in ['embedding.weight', 'out.weight', 'out.bias']:
                ckpt_tensor = cleaned[key]
                model_tensor = model.state_dict()[key].clone()
                if ckpt_tensor.dim() == 2:
                    model_tensor[:ckpt_vocab, :] = ckpt_tensor
                else:
                    model_tensor[:ckpt_vocab] = ckpt_tensor
                cleaned[key] = model_tensor
        model.load_state_dict(cleaned)
        print(f"  Checkpoint loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        uma_verified_df = None
        if os.path.exists(VERIFIED_UMA_PATH):
            uma_verified_df = pd.read_csv(VERIFIED_UMA_PATH)

        file_suffix = ''
        if args.curriculum != 'random':
            file_suffix += f"_{args.curriculum}"
        if args.suffix:
            file_suffix += f"_{args.suffix}"

        print(f"\n{'='*70}")
        print(f"EVAL-ONLY MODE (temperature={args.eval_temperature})")
        print(f"Test sets: {args.test_sets}")
        print(f"Fine bins: {args.fine_bins}, Full eval: {args.full_eval}")
        print(f"{'='*70}")
        for ts in args.test_sets:
            if ts == 'sp2013':
                ts_results = evaluate_on_sp2013(model, tokenizer, device, uma_verified_df, temperature=args.eval_temperature, fine_bins=args.fine_bins, full_eval=args.full_eval)
            else:
                ts_results = evaluate_on_test_set(model, tokenizer, device, ts, temperature=args.eval_temperature, fine_bins=args.fine_bins, full_eval=args.full_eval)
            ts_path = os.path.join(OUTPUT_DIR, f'{ts}_results{file_suffix}.csv')
            ts_results.to_csv(ts_path, index=False)
            print(f"  {ts} results saved to {ts_path}")
        return

    # Check for input files
    data_path = args.data_path if args.data_path else SYNTHETIC_TRACES_PATH
    if not os.path.exists(data_path):
        print(f"\nERROR: Training data not found at {data_path}")
        print("Run replicate_uma_paper.py first to generate traces.")
        return

    # Load synthetic traces (chunked for memory efficiency)
    print(f"\nLoading synthetic traces from {data_path}")
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    if file_size_mb > 100:  # Use chunked reading for large files
        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=100000):
            chunks.append(chunk)
        synthetic_df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
    else:
        synthetic_df = pd.read_csv(data_path)
    print(f"  Total traces: {len(synthetic_df)}")

    if args.max_samples:
        synthetic_df = synthetic_df.sample(n=min(args.max_samples, len(synthetic_df)), random_state=42)
        print(f"  Using {len(synthetic_df)} samples (subsampled)")

    print(f"  Operations: {synthetic_df['operation'].value_counts().to_dict()}")
    print(f"  Strategies: {synthetic_df['strategy'].value_counts().head(5).to_dict()}")

    # Load extra data if provided
    if args.extra_data and os.path.exists(args.extra_data):
        print(f"\nLoading extra data from {args.extra_data}")
        extra_df = pd.read_csv(args.extra_data)
        print(f"  Extra traces: {len(extra_df)}")
        print(f"  Operations: {extra_df['operation'].value_counts().to_dict()}")
        synthetic_df = pd.concat([synthetic_df, extra_df], ignore_index=True)
        print(f"  Combined total: {len(synthetic_df)} traces")

    # Load verified UMA results for comparison
    uma_verified_df = None
    if os.path.exists(VERIFIED_UMA_PATH):
        uma_verified_df = pd.read_csv(VERIFIED_UMA_PATH)
        print(f"\nLoaded verified UMA results: {len(uma_verified_df)} traces")

    # Create train/val/test splits (80/10/10)
    # First split randomly, then apply curriculum to training data only
    np.random.seed(42)
    n = len(synthetic_df)
    indices = np.random.permutation(n)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_df = synthetic_df.iloc[indices[:train_end]].reset_index(drop=True)
    val_df = synthetic_df.iloc[indices[train_end:val_end]].reset_index(drop=True)
    test_df = synthetic_df.iloc[indices[val_end:]].reset_index(drop=True)

    print(f"\nSplits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Apply data augmentation to training data
    if args.augment:
        print("\nApplying data augmentation...")
        train_df = augment_commutative(train_df)

    # Apply curriculum ordering to training data
    print(f"\nApplying curriculum: {args.curriculum}")
    train_df = apply_curriculum_ordering(train_df, args.curriculum)

    # Create datasets
    tokenizer = FractionTokenizer()
    print(f"Vocab size: {len(tokenizer)}")

    train_dataset = SyntheticTracesDataset(train_df, tokenizer, scratchpad=args.scratchpad, fine_bins=args.fine_bins)
    val_dataset = SyntheticTracesDataset(val_df, tokenizer, scratchpad=args.scratchpad, fine_bins=args.fine_bins)
    test_dataset = SyntheticTracesDataset(test_df, tokenizer, scratchpad=args.scratchpad, fine_bins=args.fine_bins)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # Build segment weights for loss masking
    segment_weights = None
    if args.weighted_loss:
        segment_weights = {'strategy': 5.0, 'goals': 3.0, 'exec': 3.0, 'work': 3.0, 'answer': 0.5, 'structural': 1.0}
        print(f"\nWeighted loss enabled: {segment_weights}")
    elif args.mask_answer:
        segment_weights = {'strategy': 1.0, 'goals': 1.0, 'exec': 1.0, 'work': 1.0, 'answer': 0.0, 'structural': 1.0}
        print(f"\nMasked answer training: answer weight = 0.0")

    collate = make_collate_fn(tokenizer, segment_weights)

    # DataLoader with optimizations for GPU
    loader_kwargs = {
        'collate_fn': collate,
        'num_workers': 4,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': True,
    }
    # Only shuffle if using random curriculum (blocked curricula must preserve order)
    shuffle_train = (args.curriculum == 'random')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_train, **loader_kwargs)
    if not shuffle_train:
        print(f"  Training with blocked curriculum (no shuffle)")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # Create model (decoder-only GPT-style)
    model = FractionGPT(
        vocab_size=len(tokenizer),
        hidden_size=args.d_model,
        nlayers=args.n_layers,
        nhead=args.n_heads,
        dropout_p=args.dropout,
        pad_idx=tokenizer.pad_token_id
    ).to(device)
    print(f"Model: Decoder-only (GPT-style)")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}")
    print(f"  layers={args.n_layers}, dropout={args.dropout}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained checkpoint if provided
    if args.load_checkpoint:
        ckpt_path = args.load_checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(SCRIPT_DIR, ckpt_path)
        print(f"\nLoading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        # Handle torch.compile() prefix
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        # Auto-detect architecture from checkpoint keys
        has_old_naming = any(k.startswith('token_embedding') for k in cleaned)
        has_cross_attn = any('multihead_attn' in k for k in cleaned)
        if has_old_naming:
            # Very old architecture (pre-3d591f3 commit fdc825e): different param names
            ckpt_vocab = cleaned['token_embedding.weight'].shape[0]
            print(f"  Detected v1 legacy checkpoint (vocab={ckpt_vocab})")
            model = LegacyFractionGPT(
                vocab_size=ckpt_vocab, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
            ).to(device)
        elif has_cross_attn:
            # Intermediate architecture: same naming, TransformerDecoder
            ckpt_vocab = cleaned['embedding.weight'].shape[0]
            print(f"  Detected v2 intermediate checkpoint (TransformerDecoder, vocab={ckpt_vocab})")
            model = FractionGPT(
                vocab_size=ckpt_vocab, hidden_size=args.d_model,
                nlayers=args.n_layers, nhead=args.n_heads,
                dropout_p=args.dropout, pad_idx=tokenizer.pad_token_id,
                legacy_decoder=True
            ).to(device)
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
        model.load_state_dict(cleaned)
        print("  Checkpoint loaded successfully")

    # Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Build file suffix for output naming
    file_suffix = ''
    if args.curriculum != 'random':
        file_suffix += f"_{args.curriculum}"
    if args.suffix:
        file_suffix += f"_{args.suffix}"

    criterion = DecoderLoss(tokenizer, pad_idx=tokenizer.pad_token_id)
    # Use fused optimizer if available (PyTorch 2.0+) with weight decay
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=(device.type == 'cuda'))
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    # Use warmup + cosine annealing for training stability
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            # Linear warmup from 0.1x to 1x
            return 0.1 + 0.9 * (epoch / args.warmup_epochs)
        else:
            # Cosine annealing after warmup
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Using LR warmup ({args.warmup_epochs} epochs) + cosine annealing")

    # Mixed precision scaler
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision training (AMP)")

    # Train
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = os.path.join(OUTPUT_DIR, f'best_model{file_suffix}.pt')
    print(f"Early stopping: patience={args.patience} epochs")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save compiled model's underlying module if compiled
            state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save(state_dict, best_model_path)
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            lr_current = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}, lr={lr_current:.2e}, no_improv={epochs_without_improvement}")

        # Early stopping check
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            print(f"Best val loss: {best_val_loss:.4f}")
            break

    # Load best model for evaluation
    best_state = torch.load(best_model_path)
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)

    # Evaluate on held-out test set
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*70}")
    print("HELD-OUT TEST SET EVALUATION")
    print(f"{'='*70}")
    print(f"Test loss: {test_loss:.4f}")

    # Evaluate on all requested test sets
    all_test_results = {}
    for ts in args.test_sets:
        if ts == 'sp2013':
            ts_results = evaluate_on_sp2013(model, tokenizer, device, uma_verified_df, temperature=args.eval_temperature, fine_bins=args.fine_bins, full_eval=args.full_eval)
        else:
            ts_results = evaluate_on_test_set(model, tokenizer, device, ts, temperature=args.eval_temperature, fine_bins=args.fine_bins, full_eval=args.full_eval)
        all_test_results[ts] = ts_results

    # Save outputs
    print(f"\n{'='*70}")
    print("SAVING OUTPUTS")
    print(f"{'='*70}")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'phase3_model{file_suffix}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.vocab,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
    }, model_path)
    print(f"  Model saved to {model_path}")

    # Save test set results
    for ts, ts_results in all_test_results.items():
        ts_path = os.path.join(OUTPUT_DIR, f'{ts}_results{file_suffix}.csv')
        ts_results.to_csv(ts_path, index=False)
        print(f"  {ts} results saved to {ts_path}")

    # Save training curves
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train', color='#1f77b4')
    plt.plot(val_losses, label='Val', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted Loss')
    plt.title(f'Phase 3: Synthetic Trace Training (d={args.d_model}, L={args.n_layers})')
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = os.path.join(OUTPUT_DIR, f'phase3_loss{file_suffix}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to {plot_path}")

    print(f"\n{'='*70}")
    print("PHASE 3 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
