
import os, sys, torch
from pathlib import Path

ROOT = Path('d:/collage/ryuu-ai')
sys.path.insert(0, str(ROOT))

from model.Ryuu_gpt import RyuuGPT
from model.config import RyuuGPTConfig
from utils.bpe_tokenizer_v2 import BPETokenizer
from Inference.prompt_builder import build_ryuu_dev_prompt

tokenizer = BPETokenizer.load(str(ROOT / 'tokenizer' / 'bpe_tokenizer_postproc.json'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompts = [
    'Define artificial intelligence in 3 short bullet points.',
    'Write a Python function to check if a list is sorted.',
    'Debug this error: TypeError: unsupported operand type(s) for +: int and str',
    'Explain difference between stack and queue with one example each.',
    'Write SQL query to find top 3 highest salaries from employees table.'
]


def extract_state_dict(data):
    if isinstance(data, dict):
        for key in ('model', 'model_state', 'state_dict'):
            if key in data and isinstance(data[key], dict):
                return data[key]
    return data


def infer_cfg_from_state(state, default_vocab_size):
    emb = state.get('token_emb.weight')
    if emb is not None and hasattr(emb, 'shape') and len(emb.shape) == 2:
        vocab_size, n_embd = int(emb.shape[0]), int(emb.shape[1])
    else:
        vocab_size, n_embd = default_vocab_size, 768

    block_ids = []
    for k in state.keys():
        if k.startswith('blocks.'):
            p = k.split('.')
            if len(p) > 1 and p[1].isdigit():
                block_ids.append(int(p[1]))
    n_layer = max(block_ids) + 1 if block_ids else 16

    rotary_key = next((k for k in state.keys() if k.endswith('attn.rotary.cos')), None)
    head_dim = int(state[rotary_key].shape[-1]) if rotary_key is not None else 64
    n_head = max(1, n_embd // max(1, head_dim))

    mask_key = next((k for k in state.keys() if k.endswith('attn.mask')), None)
    context_size = int(state[mask_key].shape[-1]) if mask_key is not None else 1024

    return RyuuGPTConfig(
        vocab_size=vocab_size,
        context_size=context_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        use_reasoning_head=any(k.startswith('reasoning_head.') for k in state.keys()),
        use_value_head=any(k.startswith('value_head.') for k in state.keys()),
    )


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = extract_state_dict(ckpt)
    cfg_dict = (ckpt.get('config') or ckpt.get('model_config')) if isinstance(ckpt, dict) else None
    cfg = RyuuGPTConfig(**cfg_dict) if cfg_dict else infer_cfg_from_state(state, tokenizer.vocab_size)
    model = RyuuGPT(cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def run_infer(model, cfg, user_prompt):
    prompt = build_ryuu_dev_prompt(user_prompt)
    ids = torch.tensor([tokenizer.encode_ids(prompt)], dtype=torch.long, device=device)
    keep = min(ids.shape[1], max(16, cfg.context_size // 2))
    ids = ids[:, -keep:]
    eos_id = tokenizer.token_to_id('<|endofturn|>')
    out = model.generate(
        ids,
        max_new_tokens=80,
        do_sample=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        eos_token_id=eos_id,
    )
    text = tokenizer.decode(out[0].tolist())
    prefix = tokenizer.decode(ids[0].tolist())
    resp = text[len(prefix):].strip() if text.startswith(prefix) else text
    return resp.encode('ascii','replace').decode('ascii')[:280]

base_path = str(ROOT / 'checkpoints' / 'v3_reasoning' / 'ckpt_best.pt')
dpo_path = str(ROOT / 'checkpoints' / 'v3_dpo' / 'ckpt_best_dpo.pt')

base_model, base_cfg = load_model(base_path)
dpo_model, dpo_cfg = load_model(dpo_path)

print('BASE:', base_path)
print('DPO :', dpo_path)
print('"''"')
for i,p in enumerate(prompts,1):
    b = run_infer(base_model, base_cfg, p)
    d = run_infer(dpo_model, dpo_cfg, p)
    print(f'[{i}] PROMPT: {p}')
    print('BASE:', b)
    print('DPO :', d)
    print('-'*80)