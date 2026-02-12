#!/usr/bin/env python3
"""
Train transformer on UMA synthetic traces.
Usage: python train_transformer.py [--epochs 50] [--batch_size 128] [--curriculum random]
"""

import os, argparse, gc, math, re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from fractions import Fraction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_PATH = os.path.join(SCRIPT_DIR, 'uma_traces_all.csv')
SP2013_PATH = os.path.join(SCRIPT_DIR, '..', 'UMA_replication', 'sp2013.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vocabulary tokens
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>', '<problem>', '</problem>',
                  '<strategy>', '</strategy>', '<answer>', '</answer>',
                  '<student>', '</student>', '<goals>', '</goals>', '<exec>', '</exec>']
DIGITS = [str(i) for i in range(1000)]
OPERATORS = ['/', '+', '-', '*', ':', '?', '→', '.']
STUDENT_PARAMS = ['g_low', 'g_mid', 'g_high', 'd_low', 'd_mid', 'd_high',
                  'rt_3', 'rt_4', 'rt_5', 'rt_6', 'ice_0', 'ice_25', 'ice_50', 'ice_75', 'ice_100']
STRATEGIES = ['KDON_AS', 'KDON_OG', 'CDON_AS', 'CDON_OG', 'ONOD_M', 'ONOD_OG', 'CROP_M', 'ICDM_D', 'ICDM_OG', 'OTHER']
RULES = ['add_fact', 'mul_fact', 'deferred_action', 'finish_problem', 'op1_none_to_zero', 'op2_none_to_zero',
         'larger_first_add_mul', 'add_to_zero', 'sub_zero_from', 'mul_by_zero', 'mul_by_one', 'acc_once',
         'acc_skip', 'acc_extra', 'acc_end', 'acc_count', 'acc_add', 'sub_fact', 'sub_LbS', 'div_calculator',
         'div_LbS', 'div_LbS_drop_rem', 'div_drop_rem', 'list_aggregation_start', 'list_aggregation_two',
         'list_aggregation_set', 'list_aggregation_calc', 'list_aggregation_skip', 'list_aggregation_zero',
         'list_aggregation_end', 'H2V_WN', 'align_right', 'VA_start', 'choose_VAS_AS', 'VA_next_calc',
         'VA_lone_carry', 'VS_next_calc', 'VS_borrow_start', 'VS_borrow_to', 'VS_borrow_from_nonzero',
         'VS_borrow_from_zero', 'VS_next_calc_swap', 'VS_borrow_from_fail', 'VAS_finish', 'VAS_end_calc',
         'VAS_do_carry', 'VAS_add_carry', 'VAS_shift_attn', 'choose_VM_M', 'VM_next_calc', 'VM_lone_carry',
         'VM_finish_sd', 'VM_end_calc', 'VM_do_carry', 'VM_add_carry', 'VM_shift1', 'VM_shift2', 'VM_new_row',
         'VM_add_parts', 'VM_shift2_no_zeros', 'ADBD_AS', 'ADBD_OG', 'ARAD_M', 'ARAD_OG', 'align_dec_no_zeros',
         'align_dec_app_zeros', 'bring_dec', 'add_dd', 'add_dd_skip', 'P2F', 'N2F_fra', 'N2F_wn', 'N2F_mix',
         'KDON_AS', 'CDON_AS', 'ONOD_M', 'ICDM_D', 'operate_nums', 'operate_dens', 'pass_den', 'convert_CD',
         'convert_CD_LCM', 'get_LCM', 'convert_fra_to_den', 'invert_op2', 'div_to_mul', 'KDON_OG', 'CDON_OG',
         'ONOD_OG', 'ICDM_OG', 'CROP_M', 'convert_CD_omit_nums', 'invert_rand', 'invert_fail',
         'div_to_mul_denied', 'check_simplify', 'skip_simplify', 'get_GCD', 'simplify_fraction', 'cannot_simplify']


def bin_param(val, thresholds, labels):
    for t, l in zip(thresholds, labels):
        if val <= t:
            return l
    return labels[-1]


def bin_student(row):
    g = bin_param(row['g'], [0.04, 0.08], ['g_low', 'g_mid', 'g_high'])
    d = bin_param(row['d'], [0.35, 0.65], ['d_low', 'd_mid', 'd_high'])
    rt = bin_param(row.get('rt_mu', 4), [3.5, 4.5, 5.5], ['rt_3', 'rt_4', 'rt_5', 'rt_6'])
    ice = bin_param(row['ice'], [12.5, 37.5, 62.5, 87.5], ['ice_0', 'ice_25', 'ice_50', 'ice_75', 'ice_100'])
    return g, d, rt, ice


class Tokenizer:
    def __init__(self):
        self.vocab = SPECIAL_TOKENS + DIGITS + OPERATORS + STUDENT_PARAMS + STRATEGIES + RULES + ['None']
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2tok = {i: t for i, t in enumerate(self.vocab)}
        self.pad_id = self.tok2id['<pad>']
        self.bos_id = self.tok2id['<bos>']
        self.eos_id = self.tok2id['<eos>']

    def encode(self, text):
        ids = [self.bos_id]
        for tok in text.split():
            if tok in self.tok2id:
                ids.append(self.tok2id[tok])
            else:
                ids.extend(self.tok2id.get(c, self.tok2id['<unk>']) for c in tok)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        skip = {'<bos>', '<eos>', '<pad>'}
        return ' '.join(self.id2tok.get(i, '<unk>') for i in ids if self.id2tok.get(i) not in skip)

    def __len__(self):
        return len(self.vocab)


class TracesDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=100):
        self.samples = []
        for _, row in df.iterrows():
            prob, resp = str(row['prob']).strip(), str(row.get('answer', '?')).strip()
            if resp in ('?', 'nan', ''):
                continue
            g, d, rt, ice = bin_student(row)
            goals = str(row['goals']) if pd.notna(row.get('goals')) else ''
            exec_r = str(row['exec']) if pd.notna(row.get('exec')) else ''

            ctx = f"<student> {g} {d} {rt} {ice} </student> <problem> {prob} </problem>"
            tgt = f"<strategy> {row['strategy']} </strategy> <goals> {goals} </goals> <exec> {exec_r} </exec> <answer> {resp} </answer>"
            full_ids = tokenizer.encode(ctx + " " + tgt)
            ctx_len = len(tokenizer.encode(ctx)) - 1
            if len(full_ids) <= max_len:
                self.samples.append((full_ids, ctx_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, ctx_len = self.samples[idx]
        return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long), ctx_len


def collate(batch):
    inputs, targets, ctx_lens = zip(*batch)
    max_len = max(len(x) for x in inputs)
    inp = torch.zeros(len(inputs), max_len, dtype=torch.long)
    tgt = torch.zeros(len(targets), max_len, dtype=torch.long)
    mask = torch.zeros(len(inputs), max_len)
    for i, (x, y, c) in enumerate(zip(inputs, targets, ctx_lens)):
        inp[i, :len(x)] = x
        tgt[i, :len(y)] = y
        mask[i, c-1:len(y)] = 1.0
    return inp, tgt, mask


def apply_curriculum(df, curriculum):
    if curriculum == 'random':
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    elif curriculum == 'blocked_complexity':
        df = df.copy()
        df['_c'] = df.apply(lambda r: len(str(r.get('goals', '')).split()) + len(str(r.get('exec', '')).split()), axis=1)
        return df.sort_values('_c').drop(columns=['_c']).reset_index(drop=True)
    elif curriculum == 'blocked_ed_first':
        def is_ud(p):
            f = re.findall(r'(\d+)/(\d+)', str(p))
            return 1 if len(f) >= 2 and int(f[0][1]) != int(f[1][1]) else 0
        df = df.copy()
        df['_ud'] = df['prob'].apply(is_ud)
        ed = df[df['_ud'] == 0].sample(frac=1, random_state=42)
        ud = df[df['_ud'] == 1].sample(frac=1, random_state=42)
        return pd.concat([ed, ud], ignore_index=True).drop(columns=['_ud'])
    elif curriculum == 'blocked_student':
        return df.copy().sort_values(['g', 'd'], ascending=[True, True]).reset_index(drop=True)
    return df


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, maxlen=5000):
        super().__init__()
        pe = torch.zeros(maxlen, d_model)
        pos = torch.arange(maxlen).unsqueeze(1)
        div = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000.) / d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1)])


class FractionGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=5, n_heads=8, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model, self.pad_idx = d_model, pad_idx
        layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1) == 1, float('-inf'))
        emb = self.pos_enc(self.embed(x))
        mem = torch.zeros(x.size(0), 1, self.d_model, device=x.device)
        return self.out(self.decoder(emb, mem, tgt_mask=mask, tgt_key_padding_mask=(x == self.pad_idx)))

    def generate(self, prompt, tokenizer, max_tokens=50, temp=1.0):
        self.eval()
        eos = tokenizer.tok2id.get('</answer>', tokenizer.eos_id)
        gen = prompt.clone()
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.forward(gen)
                next_tok = torch.multinomial(F.softmax(logits[:, -1] / temp, dim=-1), 1)
                gen = torch.cat([gen, next_tok], dim=1)
                if next_tok.item() == eos:
                    break
        return gen


def compute_loss(logits, targets, mask, pad_idx):
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=pad_idx, reduction='none').view(B, T)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, n = 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for inp, tgt, mask in loader:
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            if scaler and device.type == 'cuda':
                with autocast():
                    loss = compute_loss(model(inp), tgt, mask, model.pad_idx)
                if is_train:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss = compute_loss(model(inp), tgt, mask, model.pad_idx)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def correct_answer(prob):
    try:
        p = str(prob).replace(':', '/')
        val = eval(p.split(':')[0]) / eval(p.split(':')[1]) if ':' in str(prob) else eval(p)
        f = Fraction(val).limit_denominator(1000)
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    except:
        return '?'


def answers_match(pred, correct):
    if '?' in (pred, correct):
        return False
    try:
        def f(s):
            return float(s.split('/')[0]) / float(s.split('/')[1]) if '/' in s else float(s)
        return abs(f(pred) - f(correct)) < 1e-6
    except:
        return False


def evaluate_sp2013(model, tokenizer, device):
    model.eval()
    sp2013 = pd.read_csv(SP2013_PATH)
    configs = [(g, d, rt, ice) for g in ['g_low', 'g_mid', 'g_high']
               for d in ['d_low', 'd_mid', 'd_high'] for rt in ['rt_3', 'rt_6'] for ice in ['ice_0', 'ice_50', 'ice_100']]
    results = []
    for prob in sp2013['prob']:
        correct = correct_answer(prob)
        op = '+' if '+' in prob else '-' if '-' in prob else '*' if '*' in prob else ':'
        for g, d, rt, ice in configs:
            prompt = f"<student> {g} {d} {rt} {ice} </student> <problem> {prob} </problem>"
            ids = torch.tensor([tokenizer.encode(prompt)[:-1]], device=device)
            out = tokenizer.decode(model.generate(ids, tokenizer, 60, 0.7)[0].tolist())
            ans = out[out.find('<answer>')+8:out.find('</answer>')].strip().replace(' ', '') if '<answer>' in out and '</answer>' in out else ''
            strat = next((s for s in STRATEGIES if s in out), 'OTHER')
            results.append({'prob': prob, 'op': op, 'pred': ans, 'correct': correct,
                            'is_correct': answers_match(ans, correct), 'strategy': strat})
    df = pd.DataFrame(results)
    print(f"\nSP2013: {df['is_correct'].mean()*100:.1f}% overall")
    for op in ['+', '-', '*', ':']:
        print(f"  {op}: {df[df['op']==op]['is_correct'].mean()*100:.1f}%")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--curriculum', choices=['random', 'blocked_complexity', 'blocked_ed_first', 'blocked_student'], default='random')
    p.add_argument('--max_samples', type=int, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Curriculum: {args.curriculum}")

    # Load data
    if os.path.getsize(TRACES_PATH) > 100 * 1024 * 1024:
        df = pd.concat([c for c in pd.read_csv(TRACES_PATH, chunksize=100000)], ignore_index=True)
    else:
        df = pd.read_csv(TRACES_PATH)
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
    print(f"Traces: {len(df)}")

    # Split
    np.random.seed(42)
    idx = np.random.permutation(len(df))
    tr, va = int(0.8 * len(df)), int(0.9 * len(df))
    train_df = apply_curriculum(df.iloc[idx[:tr]].reset_index(drop=True), args.curriculum)
    val_df = df.iloc[idx[tr:va]].reset_index(drop=True)

    tokenizer = Tokenizer()
    train_ds = TracesDataset(train_df, tokenizer)
    val_ds = TracesDataset(val_df, tokenizer)
    kw = {'collate_fn': collate, 'num_workers': 4, 'pin_memory': device.type == 'cuda', 'persistent_workers': True}
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=(args.curriculum == 'random'), **kw)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=False, **kw)

    model = FractionGPT(len(tokenizer), args.d_model, args.n_layers, args.n_heads, pad_idx=tokenizer.pad_id).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    if hasattr(torch, 'compile') and device.type == 'cuda':
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = GradScaler() if device.type == 'cuda' else None

    suffix = f"_{args.curriculum}" if args.curriculum != 'random' else ''
    best_path = os.path.join(OUTPUT_DIR, f'best{suffix}.pt')
    best_loss = float('inf')
    train_losses, val_losses = [], []

    for ep in range(args.epochs):
        tl = run_epoch(model, train_ld, device, opt, scaler)
        vl = run_epoch(model, val_ld, device)
        sched.step()
        train_losses.append(tl)
        val_losses.append(vl)
        if vl < best_loss:
            best_loss = vl
            torch.save(model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(), best_path)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"Epoch {ep}: train={tl:.4f}, val={vl:.4f}")

    model.load_state_dict(torch.load(best_path))
    results = evaluate_sp2013(model, tokenizer, device)

    torch.save({'state': model.state_dict(), 'args': vars(args), 'losses': (train_losses, val_losses)},
               os.path.join(OUTPUT_DIR, f'model{suffix}.pt'))
    results.to_csv(os.path.join(OUTPUT_DIR, f'sp2013{suffix}.csv'), index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'loss{suffix}.png'), dpi=150, bbox_inches='tight')
    print("Done.")


if __name__ == '__main__':
    main()
