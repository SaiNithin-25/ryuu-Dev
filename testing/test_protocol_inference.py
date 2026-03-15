def test_protocol_prompt():
    from Inference.prompt_builder import build_ryuu_dev_prompt

    user_input = "Write a Python function to check if a number is prime."
    prompt = build_ryuu_dev_prompt(user_input)

    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    assert "expert software engineer" in prompt

    print("[OK] Protocol injection test passed")
    print(prompt[:400])


if __name__ == "__main__":
    test_protocol_prompt()
