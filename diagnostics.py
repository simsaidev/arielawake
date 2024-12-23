import os
import logging
import tweepy
import google.generativeai as genai
from datetime import datetime, timezone
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_diagnostics():
    """Run comprehensive diagnostics on the Twitter bot setup"""
    load_dotenv()
    
    results = {
        "environment_check": False,
        "api_keys_present": False,
        "twitter_auth": False,
        "gemini_auth": False,
        "tweet_conditions": False,
        "database_access": False
    }
    
    # 1. Check environment variables
    required_vars = [
        'TWITTER_API_SECRET',
        'TWITTER_API_KEY',
        'TWITTER_ACCESS_TOKEN',
        'TWITTER_ACCESS_TOKEN_SECRET',
        'TWITTER_BEARER_TOKEN',
        'GEMINI_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if not missing_vars:
        results["api_keys_present"] = True
        logging.info("✅ All required API keys are present")
    else:
        logging.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return results
    
    # 2. Test Twitter API
    try:
        client = tweepy.Client(
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        )
        
        me = client.get_me()
        if me and me.data:
            results["twitter_auth"] = True
            logging.info(f"✅ Twitter API authenticated as: @{me.data.username}")
            
            # Get recent tweets
            tweets = client.get_users_tweets(me.data.id, max_results=5)
            if tweets and tweets.data:
                latest_tweet = tweets.data[0]
                tweet_time = datetime.fromtimestamp(
                    int(latest_tweet.id >> 22) + 1288834974657,
                    timezone.utc
                )
                time_since_last = datetime.now(timezone.utc) - tweet_time
                logging.info(f"Last tweet was {time_since_last.total_seconds() / 3600:.1f} hours ago")
    except Exception as e:
        logging.error(f"❌ Twitter API test failed: {str(e)}")
        return results
    
    # 3. Test Gemini API
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Test message")
        if response and response.text:
            results["gemini_auth"] = True
            logging.info("✅ Gemini API authenticated successfully")
    except Exception as e:
        logging.error(f"❌ Gemini API test failed: {str(e)}")
        return results
    
    # 4. Check tweet conditions
    try:
        from tweet_cache import TweetCache
        cache = TweetCache(os.getenv('ENVIRONMENT', 'dev'))
        recent_tweets = cache.get_recent_tweets(limit=1)
        results["database_access"] = True
        
        if not recent_tweets:
            results["tweet_conditions"] = True
            logging.info("✅ No recent tweets found, bot should tweet")
        else:
            last_tweet_time = recent_tweets[0][3]
            hours_since_last = (datetime.now(timezone.utc) - last_tweet_time).total_seconds() / 3600
            should_tweet = hours_since_last > 1
            results["tweet_conditions"] = should_tweet
            logging.info(f"{'✅' if should_tweet else '❌'} Hours since last tweet: {hours_since_last:.1f}")
    except Exception as e:
        logging.error(f"❌ Database access test failed: {str(e)}")
        
    return results

if __name__ == "__main__":
    results = run_diagnostics()
    print("\nDiagnostics Summary:")
    for test, passed in results.items():
        print(f"{test}: {'✅ Passed' if passed else '❌ Failed'}")