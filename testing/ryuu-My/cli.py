from __future__ import annotations

from ryuu_my import RyuuMyAssistant, RyuuMyConfig


HELP_TEXT = """
Examples:
- create project UG research paper
- I need to draft the literature review by 5pm for project UG research paper #research
- remind me to send the abstract tomorrow at 9am for project UG research paper
- complete draft the literature review
- brief me
- projects
- write email to my guide about UG research paper
""".strip()


def main() -> None:
    assistant = RyuuMyAssistant(RyuuMyConfig())
    print("Ryuu-My Work Partner")
    print("Type 'help' for examples, or 'exit' to stop.")

    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            print()
            break

        lowered = user_text.lower()
        if lowered in {"exit", "quit"}:
            break
        if lowered == "help":
            print(HELP_TEXT)
            continue

        reply = assistant.handle_text(user_text)
        print(f"Ryuu-My: {reply.message}")


if __name__ == "__main__":
    main()
