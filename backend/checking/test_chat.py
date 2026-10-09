import logging
import os
import sys
import time

# Add backend folder to Python path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from chat import ChatBot  # noqa: E402


def main():
    bot = ChatBot()
    bot.warmup()

    print("=" * 50)
    print("Personal AI Chatbot  (type 'exit' to quit, 'reset' to clear memory)")
    print("=" * 50)

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            break
        if question.lower() == "reset":
            bot.reset()
            print("Memory cleared.")
            continue

        start = time.time()
        first = None

        print("\nAssistant: ", end="", flush=True)
        for piece in bot.ask_stream(question):
            if first is None:
                first = time.time() - start
            print(piece, end="", flush=True)

        print(f"\n\n[first word after {first or 0:.1f}s, "
              f"full answer after {time.time() - start:.1f}s]")


if __name__ == "__main__":
    main()