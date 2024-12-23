import os
from dotenv import load_dotenv
import yaml

def load_config(config_path=None):
    # Load environment variables from a .env file
    load_dotenv()
    
    # Load configuration from environment variables
    config = {
        'gemini': {
            'api_key': os.getenv('GEMINI_API_KEY')
        },
        'twitter': {
            'api_key': os.getenv('TWITTER_API_KEY'),
            'api_secret': os.getenv('TWITTER_API_SECRET'),
            'bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
            'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
            'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
            'user_id': os.getenv('TWITTER_USER_ID')
        },
        'bot': {
            'persona': os.getenv('BOT_PERSONA'),
            'max_tweets_per_hour': int(os.getenv('BOT_MAX_TWEETS_PER_HOUR', 4)),
            'max_responses_per_hour': int(os.getenv('BOT_MAX_RESPONSES_PER_HOUR', 10)),
            'check_interval': int(os.getenv('BOT_CHECK_INTERVAL', 300)),  # 5 minutes
            'mention_check_interval': int(os.getenv('BOT_MENTION_CHECK_INTERVAL', 300)),  # 5 minutes
            'tweet_interval': int(os.getenv('BOT_TWEET_INTERVAL', 900)),  # 15 minutes
            'mention_response_delay': int(os.getenv('BOT_MENTION_RESPONSE_DELAY', 30)),  # 30 seconds
            'discover_interval': int(os.getenv('BOT_DISCOVER_INTERVAL', 600))  # 10 minutes
        }
    }
    
    # Optionally load additional configuration from a YAML file
    if config_path:
        with open(config_path, 'r') as file:
            yaml_config = yaml.safe_load(file)
            config.update(yaml_config)
    
    # Check for critical environment variables
    critical_vars = [
        'GEMINI_API_KEY', 'TWITTER_API_KEY', 'TWITTER_API_SECRET',
        'TWITTER_BEARER_TOKEN', 'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET'
    ]
    
    for var in critical_vars:
        if not os.getenv(var):
            raise ValueError(f"Critical environment variable {var} is not set.")
    
    return config