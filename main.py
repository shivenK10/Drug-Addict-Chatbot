import sys
from typing import Dict
from logger import Logger
from emotion_detection_pipeline import detect_emotion
from generation_pipeline import generate_response

logger = Logger(
    name="AddictionChatbot",
    log_file_needed=True,
    log_file_path='Logs/chatbot.log',
    level='DEV'
)

class AddictionChatbot:
    def __init__(self):
        self.history = []

    def welcome(self):
        print("🤝 Addiction Recovery Support Bot")
        print("="*50)
        print("I’m here to help you manage cravings, prevent relapse, and cope with stress.")
        print("Type 'exit' to quit.\n")

    def chat_loop(self):
        self.welcome()
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Take care. Remember: you’re not alone in this journey. ❤️")
                break
            if not user_input:
                print("Feel free to share what’s on your mind.")
                continue

            # 1. Detect
            analysis = detect_emotion(user_input, {'history': self.history})
            print(f"[Detected: {analysis['primary_emotion'].title()}, Context: {analysis['addiction_context']}]")

            # 2. Generate
            response = generate_response(user_input, analysis)
            print(f"Bot: {response}\n")

            # 3. Log
            self.history.append({
                'input': user_input,
                'analysis': analysis,
                'response': response
            })
            if len(self.history) > 20:
                self.history = self.history[-20:]

if __name__ == "__main__":
    try:
        bot = AddictionChatbot()
        bot.chat_loop()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
