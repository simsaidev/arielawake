   # test_script.py

from main import GeminiTwitterBot  # Adjust the import based on your file structure

def main():
       # Create an instance of the bot
       bot = GeminiTwitterBot(persona_file='persona.json')

       # Test the tweet generation
       test_message = bot.generate_response()
       print("Generated Test Tweet:", test_message)

if __name__ == "__main__":
     main()