# tweet_script.py

from main import GeminiTwitterBot  # Adjust the import based on your file structure

def tweet_message():
    # Create an instance of the bot
    bot = GeminiTwitterBot()

    # Generate a tweet using the bot's capabilities
    generated_message = bot.generate_response()

    # Check if the generated message is valid
    if generated_message:
        # Post the generated tweet
        response = bot.tweet(generated_message)
        if response:
            print("Tweet posted successfully!")
        else:
            print("Failed to post tweet.")
    else:
        print("No valid tweet generated.")

if __name__ == "__main__":
    tweet_message()