# test_tweets.py
from main import GeminiTwitterBot
import logging

logging.basicConfig(level=logging.INFO)

def test_recent_tweets():
    try:
        bot = GeminiTwitterBot(persona_file='persona.json')
        tweets = bot.get_recent_tweets(limit=10)
        
        print("\nRecent Tweets:")
        for tweet_id, tweet_text in tweets:
            print(f"\nID: {tweet_id}")
            print(f"Text: {tweet_text}")
            print("-" * 50)
            
    except Exception as e:
        logging.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_recent_tweets()