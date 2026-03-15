import torch
from Inference.prompt_builder import build_ryuu_dev_prompt

def run_ryuu_dev_chat(model, tokenizer, user_text, device="cuda"):
    prompt = build_ryuu_dev_prompt(user_text)

    input_ids = tokenizer.encode_ids(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=256,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        eos_token_id=tokenizer.token_to_id("<|endofturn|>")
    )

    decoded = tokenizer.decode(output_ids[0].tolist())

    # Remove prompt prefix safely
    response = decoded[len(prompt):].strip()

    return response
