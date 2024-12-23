import os
import json
from dotenv import load_dotenv
import yaml
import logging

def load_config(config_path=None, persona_path='persona.json'):
    """
    Load configuration from environment variables and persona.json
    Args:
        config_path: Optional path to additional YAML config
        persona_path: Path to persona.json file (default: 'persona.json')
    Returns:
        dict: Complete configuration including persona settings
    """
    # Load environment variables from a .env file
    load_dotenv()
    
    # Base configuration from environment variables
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
        'database': {
            'url': os.getenv('DATABASE_URL'),
            'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', 10)),
            'timeout': int(os.getenv('DB_TIMEOUT', 30))
        },
        'bot': {
            'persona': None,  # Will be loaded from persona.json
            'max_tweets_per_hour': int(os.getenv('BOT_MAX_TWEETS_PER_HOUR', 4)),
            'max_responses_per_hour': int(os.getenv('BOT_MAX_RESPONSES_PER_HOUR', 10)),
            'check_interval': int(os.getenv('BOT_CHECK_INTERVAL', 300)),
            'mention_check_interval': int(os.getenv('BOT_MENTION_CHECK_INTERVAL', 300)),
            'tweet_interval': int(os.getenv('BOT_TWEET_INTERVAL', 900)),
            'mention_response_delay': int(os.getenv('BOT_MENTION_RESPONSE_DELAY', 30)),
            'discover_interval': int(os.getenv('BOT_DISCOVER_INTERVAL', 600))
        },
        'rate_limits': {
            'tweet_cooldown': int(os.getenv('TWEET_COOLDOWN', 900)),
            'mention_cooldown': int(os.getenv('MENTION_COOLDOWN', 60)),
            'max_retries': int(os.getenv('MAX_RETRIES', 3))
        },
        'error_handling': {
            'max_consecutive_errors': int(os.getenv('MAX_CONSECUTIVE_ERRORS', 5)),
            'error_cooldown': int(os.getenv('ERROR_COOLDOWN', 300)),
            'reset_after_errors': os.getenv('RESET_AFTER_ERRORS', 'true').lower() == 'true'
        },
        'environment': {
            'name': os.getenv('ENVIRONMENT', 'dev'),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO')
        }
    }
    
    # Load persona configuration
    try:
        with open(persona_path, 'r', encoding='utf-8') as file:
            persona_config = json.load(file)
            config['bot']['persona'] = persona_config
            logging.info(f"Loaded persona configuration for {persona_config.get('name', 'Unknown')}")
            
            # Validate required persona fields
            required_fields = ['name', 'bio', 'lore', 'rules', 'style']
            missing_fields = [field for field in required_fields if field not in persona_config]
            
            if missing_fields:
                raise ValueError(f"Missing required fields in persona config: {', '.join(missing_fields)}")
                
    except FileNotFoundError:
        raise ValueError(f"Persona configuration file not found: {persona_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in persona configuration: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading persona configuration: {str(e)}")
    
    # Optionally load additional YAML configuration
    if config_path:
        try:
            with open(config_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
                if yaml_config:
                    config.update(yaml_config)
        except Exception as e:
            logging.warning(f"Failed to load optional YAML config from {config_path}: {str(e)}")
    
    # Check for critical environment variables
    critical_vars = [
        'GEMINI_API_KEY', 
        'TWITTER_API_KEY', 
        'TWITTER_API_SECRET',
        'TWITTER_BEARER_TOKEN', 
        'TWITTER_ACCESS_TOKEN', 
        'TWITTER_ACCESS_TOKEN_SECRET'
    ]
    
    missing_vars = [var for var in critical_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing critical environment variables: {', '.join(missing_vars)}")
    
    # Validate database configuration if not in dev mode
    if config['environment']['name'] != 'dev' and not config['database']['url']:
        raise ValueError("DATABASE_URL must be set in non-dev environment")
    
    return config

def get_persona():
    """Helper function to get just the persona configuration"""
    try:
        config = load_config()
        return config['bot']['persona']
    except Exception as e:
        logging.error(f"Failed to load persona: {str(e)}")
        return None

if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        print("\nConfiguration loaded successfully!")
        print(f"Environment: {config['environment']['name']}")
        print(f"Persona: {config['bot']['persona']['name']}")
        print(f"Max tweets per hour: {config['bot']['max_tweets_per_hour']}")
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")