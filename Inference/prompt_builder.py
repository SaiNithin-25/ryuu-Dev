from pathlib import Path

PROTOCOL_PATH = Path("prompts/ryuu_dev_protocol.txt")

def load_protocol() -> str:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError("Ryuu-dev protocol file not found")
    return PROTOCOL_PATH.read_text(encoding="utf-8").strip()


def build_ryuu_dev_prompt(user_input: str) -> str:
    protocol = load_protocol()

    prompt = (
        "<|user|>\n"
        f"{protocol}\n\n"
        f"Task:\n{user_input}\n"
        "<|endofturn|>\n"
        "<|assistant|>\n"
    )

    return prompt
