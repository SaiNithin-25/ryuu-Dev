import os
import time
import glob
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from model.Ryuu_gpt import RyuuGPT
from model.config import RyuuGPTConfig
from utils.bpe_tokenizer_v2 import BPETokenizer

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
TOKENIZER_PATH = os.path.join("tokenizer", "bpe_tokenizer_postproc.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
USE_BF16 = PRECISION == "bf16"

app = FastAPI(title="Ryuu-Dev Expert API", version="1.0")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def resolve_checkpoint():
    candidates = [
        os.path.join("checkpoints", "ckpt_best.pt"),
        os.path.join("checkpoints", "test_smoke", "ckpt_best.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    step_ckpts = sorted(glob.glob(os.path.join("checkpoints", "**", "ckpt_step*.pt"), recursive=True))
    if step_ckpts:
        return max(step_ckpts, key=lambda p: int(os.path.basename(p).split("step")[-1].split(".")[0]))
    raise FileNotFoundError("No checkpoint found under checkpoints/")

# ---------------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------------
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

tokenizer = BPETokenizer.load(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size
logging.info(f"✅ Tokenizer loaded (vocab={VOCAB_SIZE})")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
MODEL_PATH = resolve_checkpoint()
logging.info(f"Loading RyuuGPT model from {MODEL_PATH} ...")
ckpt = torch.load(MODEL_PATH, map_location="cpu")

def _get_state_dict(data):
    if isinstance(data, dict):
        if "model" in data:
            return data["model"]
        if "model_state" in data:
            return data["model_state"]
        if "state_dict" in data:
            return data["state_dict"]
    return data


def _infer_cfg_from_state(state, default_vocab_size):
    emb = state.get("token_emb.weight")
    if emb is not None and hasattr(emb, "shape") and len(emb.shape) == 2:
        vocab_size, n_embd = int(emb.shape[0]), int(emb.shape[1])
    else:
        vocab_size, n_embd = default_vocab_size, 768

    block_ids = []
    for key in state.keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.append(int(parts[1]))
    n_layer = max(block_ids) + 1 if block_ids else 16

    mask_key = next((k for k in state.keys() if k.endswith("attn.mask")), None)
    context_size = int(state[mask_key].shape[-1]) if mask_key is not None else 1024

    rotary_key = next((k for k in state.keys() if k.endswith("attn.rotary.cos")), None)
    head_dim = int(state[rotary_key].shape[-1]) if rotary_key is not None else 64
    n_head = max(1, n_embd // max(1, head_dim))

    return RyuuGPTConfig(
        vocab_size=vocab_size,
        context_size=context_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        use_reasoning_head=any(k.startswith("reasoning_head.") for k in state.keys()),
        use_value_head=any(k.startswith("value_head.") for k in state.keys()),
    )

state = _get_state_dict(ckpt)
cfg_dict = ckpt.get("config") or ckpt.get("model_config") or {} if isinstance(ckpt, dict) else {}
config = RyuuGPTConfig(**cfg_dict) if cfg_dict else _infer_cfg_from_state(state, VOCAB_SIZE)
model = RyuuGPT(config).to(DEVICE)
model.load_state_dict(state, strict=False)
model.eval()

if USE_BF16:
    logging.info("✅ Using BF16 precision for inference")
else:
    logging.info("⚡ Using FP16 precision for inference")

logging.info("✅ Model ready for inference")

# ---------------------------------------------------------
# INPUT/OUTPUT MODELS
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True

class QueryResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    generation_time: float
    tokens_per_sec: float
    generated_text: str

# ---------------------------------------------------------
# HELPER: Generate with timing + metrics
# ---------------------------------------------------------
@torch.inference_mode()
def generate_response(prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float, do_sample: bool):
    inputs = tokenizer.encode_ids(prompt)
    input_ids = torch.tensor(inputs, dtype=torch.long, device=DEVICE).unsqueeze(0)
    input_len = input_ids.shape[1]

    logging.info(f"🧠 Generating for input ({input_len} tokens)...")

    start_time = time.time()
    if DEVICE == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if USE_BF16 else torch.float16):
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
        torch.cuda.synchronize()
    else:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )
    end_time = time.time()

    output_len = output_ids.shape[1]
    total_new = output_len - input_len
    gen_time = end_time - start_time
    speed = total_new / gen_time if gen_time > 0 else 0.0

    decoded = tokenizer.decode(output_ids[0].tolist())

    return {
        "input_tokens": input_len,
        "output_tokens": total_new,
        "total_tokens": output_len,
        "generation_time": round(gen_time, 3),
        "tokens_per_sec": round(speed, 2),
        "generated_text": decoded,
    }

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "🚀 Ryuu-Dev Expert API is online."}

@app.post("/query", response_model=QueryResponse)
async def query_model(req: QueryRequest):
    try:
        result = generate_response(
            prompt=req.query,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            do_sample=req.do_sample
        )
        return result
    except Exception as e:
        logging.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
