from fastapi import FastAPI
import google.generativeai as genai
import tweepy
import json
from datetime import datetime, timedelta, timezone
import logging
import time
from config import load_config  # Import the configuration loader
import os  # Add this import statement
from dotenv import load_dotenv  # Add this import statement
from tweet_cache import TweetCache  # Import the TweetCache class
from learning_store import LearningStore  # Add this line with other imports
import asyncio  # Add this import statement
import random
from rate_limiter import RateLimiter
from unidecode import unidecode

    
app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

class GeminiTwitterBot:
    def __init__(self, persona_file='persona.json'):
        """Initialize the Twitter bot with proper configuration and state management"""
        # Load environment variables
        load_dotenv()
        
        # Determine environment
        self.environment = os.getenv('ENVIRONMENT', 'dev')
        
        try:
              # Load config and persona
            self.config = load_config()
            self.persona = self.config['bot']['persona']
            
            # Environment
            self.environment = self.config['environment']['name']
            
            # API credentials from config
            twitter_config = self.config['twitter']
            self.api_key = twitter_config['api_key']
            self.api_key_secret = twitter_config['api_secret']
            self.access_token = twitter_config['access_token']
            self.access_token_secret = twitter_config['access_token_secret']
            self.bearer_token = twitter_config['bearer_token']
            self.gemini_key = self.config['gemini']['api_key']
            
            # Initialize timing configuration from config
            self.max_tweets_per_hour = self.config['bot']['max_tweets_per_hour']
            self.max_responses_per_hour = self.config['bot']['max_responses_per_hour']
            self.check_interval = self.config['bot']['check_interval']
            self.mention_check_interval = self.config['bot']['mention_check_interval']
            self.tweet_interval = self.config['bot']['tweet_interval']
            self.mention_response_delay = self.config['bot']['mention_response_delay']
            self.discover_interval = self.config['bot']['discover_interval']
            
            # Initialize components
            self.tweet_cache = TweetCache(self.environment)
            self.learning_store = LearningStore()
            self.rate_limiter = RateLimiter()
            
            # Initialize state tracking
            self._last_tweet_time = None
            self._last_mention_check = None
            self._last_discovery = None
            self._last_cleanup = None
            self.is_configured = False
            
            # Initialize APIs
            self._init_apis()
            
            logging.info(f"Bot initialized for {self.persona['name']} in {self.environment} environment")
            
            # Set up retries
            self.max_retries = 3
            self.retry_delay = 5
            
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            raise
        
    def _init_apis(self):
        """Initialize Twitter and Gemini APIs"""
        try:
            # Initialize Twitter client with authentication
            self.Client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_key_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret
            )

            # Initialize Gemini
            genai.configure(api_key=self.gemini_key)
            self.model = genai.GenerativeModel('gemini-pro')

            # Verify Twitter authentication with retries
            max_retries = 3
            retry_delay = 60

            for attempt in range(max_retries):
                try:
                    me = self.Client.get_me(user_auth=True)
                    if me and me.data:
                        self.bot_user_id = me.data.id
                        self.bot_username = me.data.username
                        self.is_configured = True
                        logging.info(f"Authenticated as @{self.bot_username}")
                        break
                except tweepy.TweepyException as e:
                    if e.response is not None and e.response.status_code == 429:
                        reset_time = int(e.response.headers.get('X-Rate-Limit-Reset', 0))
                        retry_after = max(reset_time - int(time.time()), retry_delay)
                        logging.warning(f"Rate limit exceeded during authentication. Waiting for {retry_after} seconds before retrying (attempt {attempt + 1}/{max_retries}).")
                        time.sleep(retry_after)
                    else:
                        raise
            else:
                raise ValueError("Failed to authenticate with Twitter after multiple retries")

        except Exception as e:
            logging.error(f"API initialization error: {e}")
            raise
        
    def get_rate_limit_status(self):
        """Get current rate limit status from rate limiter."""
        limiter_status = self.rate_limiter.get_status()
        current_time = datetime.now(timezone.utc)
        time_since_last = None

        if self._last_tweet_time:
            time_since_last = (current_time - self._last_tweet_time).total_seconds()

        tweet_status = limiter_status.get('tweet', {})
        
        return {
            'current_time': current_time.isoformat(),
            'tweets': {
                'last_hour': tweet_status.get('used_this_hour', 0),
                'max_per_hour': tweet_status.get('max_per_hour', self.max_tweets_per_hour),
                'remaining': tweet_status.get('remaining', 0),
                'time_since_last': time_since_last,
                'min_interval': self.tweet_interval,
                'next_available': tweet_status.get('rate_limit_reset')
            },
            'responses': {
                'max_per_hour': self.max_responses_per_hour,
                'min_delay': self.mention_response_delay,
                'used_this_hour': limiter_status.get('reply', {}).get('used_this_hour', 0)
            },
            'intervals': {
                'check': self.check_interval,
                'mention_check': self.mention_check_interval,
                'discover': self.discover_interval
            },
            'can_tweet_now': self._can_tweet_now(limiter_status)
        }

        
    def get_recent_tweets(self, username=None, limit=10):
        """Get recent tweets - dev mode fallback"""
        if self.environment == 'dev':
            # Return empty list or mock data for development
            return [{'text': 'Test tweet', 'id': '1', 'timestamp': datetime.now(timezone.utc)}]
        
        try:
            tweets = self.tweet_cache.get_recent_tweets(limit=limit)
            return [{
                'id': tweet[0],
                'text': tweet[1], 
                'url': tweet[2],
                'timestamp': tweet[3]
            } for tweet in tweets]
        except Exception as e:
            logging.error(f"Error retrieving tweets: {e}")
            return []
    
    def _make_twitter_request(self, request_func):
        """Simplified Twitter API request handler with Tweepy rate limiting"""
        try:
            return request_func()
        except Exception as e:
            logging.error(f"Request failed: {e}")
            return None
        
    def load_persona(self, file_path):
        """Load and validate persona configuration"""
        try:
            with open(file_path, 'r') as file:
                persona_data = json.load(file)
                
            # Validate required persona fields
            required_fields = ['rules', 'bio', 'lore']
            missing_fields = [field for field in required_fields 
                            if not persona_data.get(field)]
            
            if missing_fields:
                raise ValueError(f"Missing required persona fields: {missing_fields}")
                
            logging.info("Persona loaded successfully")
            return persona_data
            
        except FileNotFoundError:
            logging.error(f"Persona file not found: {file_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in persona file: {file_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading persona: {e}")
            raise

    def generate_image_prompt(self, tweet_text):
        """Generate a simple, clean image prompt based on the tweet content"""
        prompt = f"""
        Create a simple visual description in plain English:
        - Describe a single scene or image
        - Absolutely NO ASCII art or special characters
        - No technical symbols or formatting
        - Use simple descriptive language
        - Focus on visual elements only
        
        Generate an image prompt based on the tweet content:
        
        Context for what image should mainly be composed about: {tweet_text}
        
        Use your image style: {self.persona.get('image', {}).get('style')}
        
        Colors to use: {self.persona.get('image', {}).get('colors')}
        
        Bio for reference: {self.persona.get('bio', {})}
        
        Lore for reference: {self.persona.get('lore', {})}
        
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'top_k': 40,
                    'top_p': 0.9,
                }
            )
            if response:
                # Properly encode and decode the text
                clean_prompt = response.text.strip().encode('utf-8').decode('utf-8')
                logging.info(f"Generated clean image prompt: {clean_prompt}")
                return clean_prompt
                
        except Exception as e:
            logging.error(f"Image prompt generation failed: {e}")
            return None

    def generate_image(self, prompt):
        try:
            # Use Gemini to clean the prompt
            clean_response = self.model.generate_content(
                f"Clean the following text by removing ASCII art, special characters, and formatting. Return only plain descriptive text suitable for DALL-E: {prompt}",
                generation_config={
                    'temperature': 0.1,
                    'top_k': 1,
                    'top_p': 0.1,
                }
            )
            
            cleaned_prompt = clean_response.text.strip()
            logging.info(f"Original prompt: {prompt}")
            logging.info(f"Gemini-cleaned prompt: {cleaned_prompt}")
            
            try:
                # Now use the cleaned prompt for DALL-E
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=cleaned_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
            except UnicodeEncodeError:
                logging.warning("Encountered UnicodeEncodeError, retrying with ASCII prompt")
                cleaned_prompt = unidecode(cleaned_prompt)
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=cleaned_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
            
            if not response or not response.data:
                logging.error("No response from DALL-E")
                return None
                    
            # Get the image URL
            image_url = response.data[0].url
            
            # Download the image
            import requests
            from tempfile import NamedTemporaryFile
            
            r = requests.get(image_url)
            if r.status_code == 200:
                with NamedTemporaryFile(delete=False, suffix=".png") as f:
                    f.write(r.content)
                    return f.name
                    
            return None
            
        except Exception as e:
            logging.error(f"Image generation failed: {e}", exc_info=True)
            return None
        
    def generate_response(self, prompt=None, context=None):
        """Generate unique responses with improved context awareness"""
        
        # Get learning data
        system_prompt = self.config['bot']['persona']
        rules = "\n".join(self.persona.get("rules", []))
        bio = "\n".join(self.persona.get("bio", []))
        lore = "\n".join(self.persona.get("lore", []))
        random_actions = "\n".join(self.persona.get("style", {}).get("random_actions", []))
        style = "\n".join(self.persona.get("style", {}).get("all", []))
        friends = "\n".join(self.persona.get("relationships", {}).get("friends", []))
        agents = "\n".join(self.persona.get("relationships", {}).get("agents", []))
        # Add pre-prompt instruction about hashtags
        pre_prompt = f"""
        IMPORTANT: Generate responses in plain text only. DO NOT add any hashtags, 
        do not add # symbols anywhere, do not add tags at the end of the response, 
        and do not mark keywords with #. The response must be pure text. Remember your name is `{self.persona.get('name')} Make sure your response is less than 280 characters and never cut off your sentence`.
        
        Examples of what NOT to do:
        - "Great insight about AI #AI #Tech"
        - "The future is here... #AIthoughts"
        - Adding any hashtags at the end of tweets
        - Adding any links or links in the response
        """
        
        # Enhanced prompt engineering
        full_prompt = f"""
        {pre_prompt}
        
        {system_prompt}
        
        Bio (for reference of past only):
        {bio}
        
        Lore (for reference of past only):
        {lore}
        
        Style:
        {style}
        
        Core Rules:
        {rules}
        
        Friends (dont mention them in your tweets but they are your friends and they are here to help you):
        {friends}
        
        Agents (dont mention them in your but they are other agents like you):
        {agents}
        
        Random actions:
        {random_actions}
        
        Context Analysis:
            {context if f'Answer to this context: {prompt}.'  else 'This is a new thought and context is not provided.'}
            
        Thought Evolution Instructions:
            1. Be witty, creative, unique, find meanings in persons words and metaphoric without cryptic
            2. Dont be cheesy, be natural and human
            3. Build on successful interaction patterns but evolve them further
            4. Reference past insights while pushing new boundaries
            5. Create unexpected connections between tech concepts
            6. Question established paradigms provocatively
            7. Maintain your persona while being accessible
            8. Generate a tweet that uses plain text only - absolutely NO hashtags (#), NO tickers ($), NO emoji, NO links, and NO promotional content. 
            8. Focus on expressing the idea directly
            
        Task: {prompt if f'Answer this prompt with less than 280 characters and a clear, direct response, without cutting off your sentence. DO NOT USE hashtags (#) or tickers ($) or promote a project or cryptocurrency in general. Never use hashtags, tickers, emoji, links, or promotional content: {prompt}. Also you dont have to mention context just answer with the context in mind, the prompt is the main thing to focus on.'  else 'Generate a random thought, direct insight that challenges current paradigms without being cryptic and is less than 280 characters, without cutting off your sentence. Be witty, creative, unique, and stay true to your persona without being cryptic. Your goal isnt to recite facts about yourself but use the provided knowledge to create a new thought. DO NOT USE hashtags (#) or tickers ($) or promote a project or cryptocurrency in general.'}
        """
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0.9,
                    'top_k': 40,
                    'top_p': 0.9,
                },
                safety_settings={
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", 
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                }
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from model")

            # Enhanced response cleaning
            final_response = response.text.strip()
            
            # Remove any hashtags or marked words
            final_response = ' '.join(word for word in final_response.split() 
                                    if not word.startswith('#')
                                    and not word.startswith('$')
                                    and not word.endswith('#')
                                    and '#' not in word)
                                    
            final_response = final_response.replace('#', '')
            final_response = final_response[:280]
                
            # Enhanced uniqueness analysis
            uniqueness_score = self.analyze_tweet_uniqueness(final_response)
            creativity_score = self.analyze_creativity(final_response)
            
            # Only store truly unique content
            if uniqueness_score > 7 and creativity_score:
                self.learning_store.store_interaction(
                    text=prompt,
                    response=final_response,
                    uniqueness_score=uniqueness_score,
                    creativity_notes=creativity_score,
                    evolution_analysis=f"Uniqueness: {uniqueness_score}/10"
                )
            
            return final_response
            
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            return None

    def finalize_tweet_with_gemini(self, message):
        """Enhanced tweet validation with better logging"""
        if not message:
            logging.error("Empty message received for validation")
            return None
            
        logging.info(f"Validating tweet: {message}")
        
        # Basic validation first
        if any(char in message for char in ['#', '$', '🔥', '💫', '🚀']):
            logging.warning(f"Tweet contains banned characters, cleaning up: {message}")
            # Clean instead of reject
            cleaned = ''.join(c for c in message if c not in ['#', '$'])
            message = cleaned
        
        # Ensure message isn't too long
        if len(message) > 280:
            message = message[:280]
            logging.info("Tweet truncated to 280 characters")
        
        # Simplified validation prompt
        prompt = f"""
        You are a tweet validator. Your job is to check if this tweet follows the rules and return either:
        1. The cleaned tweet text if it's okay (with any needed minor fixes like removing hashtags or tickers)
        2. The exact string "REVIEW_REQUIRED" if the tweet breaks rules
        
        Rules to check:
        - Keep language in tact no need to correct grammar or slang
        - No hashtags (#)
        - No ticker symbols ($)
        - No emoji
        - No links
        - No promotional content
        - Must be clear and direct
        - Keep original meaning intact
        
        Tweet to validate: {message}
        
        Return ONLY the cleaned tweet or "REVIEW_REQUIRED". No other text or explanation.
        """
        
        try:
            response = self.model.generate_content(prompt,
                generation_config={
                    'temperature': 0.1,  # Lower temperature for more consistent validation
                    'top_k': 1,
                    'top_p': 0.1,
                })
            
            if not response or not response.text:
                logging.error("Empty response from Gemini validation")
                return message  # Return original if validation fails
                
            validated_text = response.text.strip()
            
            if "REVIEW_REQUIRED" in validated_text:
                logging.warning(f"Tweet rejected by validator: {message}")
                return None
                
            # Final safety checks
            final_text = validated_text[:280]
            logging.info(f"Tweet validated successfully: {final_text}")
            return final_text
            
        except Exception as e:
            logging.error(f"Gemini validation error: {e}")
            # If validation fails, return original message rather than blocking
            return message

    def handle_mentions(self, since_id=None):
        """Handle mentions with since_id support"""
        logging.info("Handling mentions...")
        
        # Check rate limits first
        can_mention, rate_message = self.rate_limiter.check_limit("mention")
        if not can_mention:
            logging.warning(f"Rate limit check failed: {rate_message}")
            return
        
        logging.info("Rate limit check passed, proceeding with mention handling")
            
        try:
            # Get authenticated user info
            me = self._make_twitter_request(lambda: self.Client.get_me(user_auth=True))
            if not me or not me.data:
                logging.error("Could not get user data")
                return
                
            user_id = me.data.id
            
            mentions_params = {
                'id': user_id,
                'expansions': ['referenced_tweets.id'],
                'max_results': 20,
                'tweet_fields': ['author_id', 'conversation_id', 'created_at', 'text'],
                'user_auth': True
            }
            
            # Use since_id if provided, otherwise default to time-based
            if since_id:
                mentions_params['since_id'] = since_id
            
            # Get mentions
            mentions = self._make_twitter_request(
                lambda: self.Client.get_users_mentions(**mentions_params)
            )

            if not mentions or not mentions.data:
                logging.info("No new mentions found")
                return

            for mention in mentions.data:
                # Check rate limits for each mention
                can_mention, rate_message = self.rate_limiter.check_limit("mention")
                if not can_mention:
                    logging.info(f"Rate limit reached during mention processing: {rate_message}")
                    break
                    
                # Skip if already processed
                if self.tweet_cache.has_processed_tweet(mention.id):
                    logging.info(f"Skipping mention {mention.id} - already processed")
                    continue

                try:
                    # Get conversation context
                    context = self._get_conversation_context(mention)

                    # Check if we should respond
                    if self.should_respond(mention):
                        response = self.generate_response(
                            prompt=f"Respond to mention: {mention.text}",
                            context=context
                        )
                        
                        if response:
                            # Post reply
                            reply = self._make_twitter_request(lambda: self.Client.create_tweet(
                                text=response[:280],
                                in_reply_to_tweet_id=mention.id,
                                user_auth=True
                            ))
                            
                            if reply and reply.data:
                                # Record the successful mention response
                                self.rate_limiter.record_action("mention")
                                
                                # Store mention and response
                                self.tweet_cache.add_mention(mention.id, mention.text, response)
                                
                                # Store interaction data
                                self.learning_store.store_interaction(
                                    text=mention.text,
                                    response=response,
                                    conversation_id=mention.conversation_id
                                )
                                
                                logging.info(f"Successfully replied to mention {mention.id}")
                                
                                # Mark mention as processed
                                self.tweet_cache.mark_tweet_processed(mention.id)
                            else:
                                logging.error("Failed to post reply to mention")
                    else:
                        logging.info(f"Skipping mention {mention.id} - response criteria not met")
                    
                    # Mark mention as processed even if not responded to
                    self.tweet_cache.mark_tweet_processed(mention.id)
                    
                except Exception as mention_error:
                    logging.error(f"Error processing mention {mention.id}: {str(mention_error)}")
                    self.tweet_cache.mark_tweet_processed(mention.id)  # Mark as processed even if error
                    continue  # Continue to next mention
                        
        except Exception as e:
            logging.error(f"Error handling mentions: {e}")
            logging.exception("Full traceback:")
            
    def _get_conversation_context(self, mention):
        """Get full conversation context for a mention"""
        context = []
        
        try:
            # Get parent tweet if this is a reply
            if mention.referenced_tweets:
                for ref in mention.referenced_tweets:
                    if ref.type == 'replied_to':
                        parent = self._make_twitter_request(
                            lambda: self.Client.get_tweet(
                                ref.id,
                                tweet_fields=['text', 'author_id', 'created_at']
                            )
                        )
                        if parent and parent.data:
                            context.append(f"Parent tweet: {parent.data.text}")

            # Add the mention text
            context.append(f"Current mention: {mention.text}")
            
            return "\n".join(context)
            
        except Exception as e:
            logging.error(f"Error getting conversation context: {e}")
            return mention.text  # Fallback to just the mention text

    def handle_engagement_tracking(self):
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            # Get tweets as tuples from database
            recent_tweets = self.tweet_cache.get_recent_tweets(limit=100)
            tweets_to_check = [
                t for t in recent_tweets 
                if t[3] > cutoff_time  # Index 3 is timestamp in tuple
            ]

            for tweet in tweets_to_check:
                tweet_metrics = self.Client.get_tweet(
                    tweet[0],  # Index 0 is tweet ID
                    tweet_fields=['public_metrics'],
                    user_auth=True
                )
                
                if tweet_metrics and tweet_metrics.data:
                    metrics = tweet_metrics.data.public_metrics
                    engagement_score = (
                        metrics['like_count'] * 2 +
                        metrics['retweet_count'] * 3 +
                        metrics['reply_count']
                    )
                    
                    uniqueness_score = self.analyze_tweet_uniqueness(tweet[1])  # Index 1 is tweet text
                    creativity_notes = self.analyze_creativity(tweet[1])
                    
                    # Store learning data
                    logging.info(f"Storing learning data for tweet {tweet[0]}")
                    self.learning_store.store_interaction(
                        text=tweet[1],  # Index 1 is tweet text
                        response=None,
                        engagement=engagement_score,
                        uniqueness_score=uniqueness_score,
                        creativity_notes=creativity_notes,
                        timestamp=tweet[3]  # Index 3 is timestamp
                    )
                    
                    logging.info(f"Engagement tracking complete for tweet {tweet[0]}")

        except Exception as e:
            logging.error(f"Error checking engagements: {e}")

    def analyze_creativity(self, text):
        """Analyze creativity through your perspective"""
        analysis_prompt = f"""
        You are {self.persona.get('name')}, and your persona is {self.persona.get('bio')}.
        
        Analyze why this tweet is relevent to you
        
        Tweet: {text}
        
        Identify how it demonstrates:
        {self.persona.get('topics')}
        
        Focus on what makes it uniquely valuable to you.
        Return brief bullet points only for successful elements.
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            return response.text
        except:
            return None

    def analyze_tweet_uniqueness(self, tweet_text):
        """Enhanced uniqueness analysis with repetition detection"""
        tweets = self.tweet_cache.get_recent_tweets(limit=10)
        recent_tweets = [t[1] for t in tweets] # Index 1 is text column
        
        # Check for similar content using basic similarity metrics
        for past_tweet in recent_tweets:
            if self._calculate_similarity(tweet_text, past_tweet) > 0.7:  # 70% similarity threshold
                return 0  # Force rejection of similar tweets
        
        analysis_prompt = f"""
        Analyze this tweet's uniqueness (0-10):

        New tweet: {tweet_text}
        Recent tweets: {json.dumps(recent_tweets, indent=2)}
        
        Score (0-10) for:
        1. Vocabulary uniqueness
        2. Novel tech concepts
        3. Original perspective
        4. Thought evolution
        5. Provocative elements
        
        Return only comma-separated scores
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            scores = [int(s) for s in response.text.strip().split(',')]
            if len(scores) == 5:
                weighted_score = (
                    scores[0] * 0.2 +   # Increased weight on vocabulary
                    scores[1] * 0.25 +  # Tech concepts
                    scores[2] * 0.2 +   # Perspective
                    scores[3] * 0.2 +   # Evolution
                    scores[4] * 0.15    # Provocative
                )
                return round(weighted_score, 1)
        except Exception as e:
            logging.error(f"Uniqueness analysis failed: {e}")
        return 5

    def _calculate_similarity(self, text1, text2):
        """Calculate basic text similarity score"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)
    
    def _can_tweet_now(self, limiter_status):
        """Check if tweeting is currently allowed."""
        if not self.is_configured:
            return False

        # Check rate limiter status
        tweet_status = limiter_status.get('tweet', {})
        if tweet_status.get('is_limited', False):
            return False

        # Check tweets per hour
        if tweet_status.get('used_this_hour', 0) >= self.max_tweets_per_hour:
            return False

        # Check minimum interval
        if self._last_tweet_time:
            time_since_last = (datetime.now(timezone.utc) - self._last_tweet_time).total_seconds()
            if time_since_last < self.tweet_interval:
                return False

        return True

    def should_tweet(self):
        """Determine if tweeting is currently allowed"""
        # Get rate limiter status
        limiter_status = self.rate_limiter.get_status()
        
        # First check if we can tweet based on limits
        if not self._can_tweet_now(limiter_status):
            return False
        
        # Only check API health if limits allow tweeting
        try:
            # Cache API check result briefly
            if not hasattr(self, '_last_api_check') or \
            (datetime.now(timezone.utc) - self._last_api_check).total_seconds() > 60:
                self._last_api_check = datetime.now(timezone.utc)
                test_call = self._make_twitter_request(lambda: self.Client.get_me())
                if not test_call:
                    return False
        except Exception as e:
            logging.warning(f"API health check failed: {e}")
            return False
            
        return True

    def tweet(self, message=None):
        if not self.should_tweet():
            return False
            
        try:
            if not message:
                message = self.generate_response()
                if not message:
                    return False
                    
            message = self.finalize_tweet_with_gemini(message)
            if not message:
                return False
                
            media_id = None
            if random.random() < 0.1:
                image_prompt = self.generate_image_prompt(message)
                if image_prompt:
                    logging.info(f"Generated image prompt: {image_prompt}")
                    image_path = self.generate_image(image_prompt)
                    
                    if image_path:
                        # Create uploader if not exists
                        if not hasattr(self, 'uploader'):
                            auth = tweepy.OAuth1UserHandler(
                                self.api_key, self.api_key_secret,
                                self.access_token, self.access_token_secret
                            )
                            self.uploader = tweepy.API(auth)

                        # Upload image
                        media = self.uploader.media_upload(filename=image_path)
                        media_id = media.media_id
                        
                        # Clean up the temporary file
                        try:
                            os.unlink(image_path)
                        except Exception as e:
                            logging.error(f"Error cleaning up image file: {e}")
            
            # Post tweet with optional media
            tweet_params = {"text": message}
            if media_id:
                tweet_params["media_ids"] = [media_id]
                
            response = self._make_twitter_request(
                lambda: self.Client.create_tweet(**tweet_params)
            )
            
            if response and hasattr(response, 'data'):
                self._last_tweet_time = datetime.now(timezone.utc)
                self.rate_limiter.record_action("tweet")
                
                tweet_id = response.data['id']
                tweet_url = f"https://twitter.com/{self.bot_username}/status/{tweet_id}"
                
                self.tweet_cache.add_tweet(tweet_id, message, tweet_url)
                logging.info(f"Tweet posted successfully: {tweet_url}")
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error posting tweet: {e}")
            return False

    
    def should_respond(self, mention):
        """Determine if we should respond to a mention"""
        try:
            logging.info(f"Determining if should respond to mention {mention.id}")
            
            # Check if already processed
            if self.tweet_cache.has_processed_mention(mention.id) or self.tweet_cache.has_processed_reply(mention.id):
                logging.info(f"Skipping mention {mention.id} - already processed as mention or reply")
                return False
            
            # Check rate limits
            can_mention, _ = self.rate_limiter.check_limit("mention")
            if not can_mention:
                logging.info(f"Skipping mention {mention.id} - rate limit reached")
                return False
                
            # Skip if we don't have mention text
            if not mention or not mention.text:
                return False
                
            # Extract mention text without @mentions
            mention_text = ' '.join(word for word in mention.text.split() 
                                if not word.startswith('@'))
                                
            # Skip if there's no actual content after removing @mentions
            if not mention_text.strip():
                return False
                
            # Skip if tweet is too short
            if len(mention_text.split()) < 3:
                return False
                
            # Skip if it contains spam indicators
            spam_indicators = [
                'giveaway', 'follow me', 'check out', 'win', 'contest',
                'airdrop', 'whitelist', 'link in bio', 'dm me'
            ]
            if any(indicator in mention_text.lower() for indicator in spam_indicators):
                return False
                
            # Skip if it's just emojis or basic reactions
            basic_reactions = ['nice', 'cool', 'good', 'wow', 'lol', 'great']
            words = [w.lower() for w in mention_text.split()]
            if all(word in basic_reactions for word in words):
                return False
                
            # Check if the content is meaningful
            if not self.has_meaningful_content(mention_text):
                return False
                
            # Additional context analysis using Gemini
            analysis_prompt = f"""
            Analyze if this mention warrants a response (YES/NO):
            Text: {mention_text}
            
            Criteria:
            1. Shows genuine engagement or asks a question
            2. Relates to tech, AI, development, or your interests
            3. Has substance beyond basic reactions
            4. Could lead to meaningful discussion
            5. Not spam or promotional
            
            Return ONLY YES or NO.
            """
            
            try:
                response = self.model.generate_content(
                    analysis_prompt,
                    generation_config={
                        'temperature': 0.1,
                        'top_k': 1,
                        'top_p': 0.1,
                    }
                )
                
                should_respond = 'YES' in response.text.upper()
                logging.info(f"Should respond to mention {mention.id}: {should_respond}")
                return should_respond
                
            except Exception as e:
                logging.error(f"Error in content analysis: {e}")
                # Fallback to simpler check
                has_question = '?' in mention_text
                has_keywords = any(word in mention_text.lower() for word in 
                                ['why', 'how', 'what', 'when', 'ai', 'tech', 'code', 'build'])
                return has_question or has_keywords
                
        except Exception as e:
            logging.error(f"Error in should_respond: {e}")
            return False
        
    def has_meaningful_content(self, text):
        """Enhanced content analysis for replies"""
        analysis_prompt = f"""
        Analyze if this reply warrants a response (answer YES or NO only):
        Reply text: {text}
        
        Criteria:
        1. Contains a question or seeks information
        2. Discusses {self.persona.get('topics')}
        3. Offers substantive thoughts or insights
        4. Not just simple reactions or emojis
        5. Could lead to meaningful discussion
        6. Not spam or promotional content
        7. Shows genuine engagement with the topic
        """
        
        try:
            response = self.model.generate_content(
                analysis_prompt,
                generation_config={
                    'temperature': 0.1,
                    'top_k': 1,
                    'top_p': 0.1,
                }
            )
            return 'YES' in response.text.upper()
        except Exception as e:
            logging.error(f"Content analysis failed: {e}")
            # Fallback to basic keyword checking
            meaningful_topics = [
                topic for topic in self.persona.get('topics')
            ]
            return any(topic in text.lower() for topic in meaningful_topics)

    def handle_tweet_replies(self):
        logging.info("Handling tweet replies...")
        
        # Check rate limits first
        can_reply, rate_message = self.rate_limiter.check_limit("reply")
        if not can_reply:
            logging.warning(f"Rate limit check failed: {rate_message}")
            return
        
        logging.info("Rate limit check passed, proceeding with reply handling")

        try:
            # Get bot's user ID once
            me = self._make_twitter_request(lambda: self.Client.get_me(user_auth=True))
            if not me or not me.data:
                logging.error("Could not get user info")
                return
                
            bot_user_id = me.data.id
            
            since_time = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_tweets = self._make_twitter_request(lambda: self.Client.get_users_tweets(
                id=bot_user_id,
                start_time=since_time,
                max_results=20,
                tweet_fields=['conversation_id'],
                user_auth=True
            ))

            for tweet in recent_tweets:
                # Check rate limits for each tweet's replies
                can_reply, rate_message = self.rate_limiter.check_limit("reply")
                if not can_reply:
                    logging.info(f"Rate limit reached during reply processing: {rate_message}")
                    break
                    
                try:
                    # Search for replies to this tweet
                    reply_query = f"conversation_id:{tweet['id']} -from:{bot_user_id}"
                    replies = self._make_twitter_request(lambda: self.Client.search_recent_tweets(
                        query=reply_query,
                        max_results=10,  # Reduced for efficiency  
                        expansions=['referenced_tweets.id'],
                        tweet_fields=['author_id', 'conversation_id', 'created_at', 'text'],
                        user_auth=True
                    ))
                    
                    if not replies or not replies.data:
                        continue
                    
                    # Process each reply
                    for reply in replies.data:
                        # Check rate limits again for each individual reply
                        can_reply, rate_message = self.rate_limiter.check_limit("reply")
                        if not can_reply:
                            logging.info(f"Rate limit reached during individual reply: {rate_message}")
                            break

                        try:
                            # Skip if already processed
                            if self.tweet_cache.has_processed_tweet(reply.id):
                                logging.info(f"Skipping reply {reply.id} - already processed")
                                continue
                                
                            # Get reply context  
                            context = self._get_reply_context(reply)
                            
                            # Check if content is worth responding to
                            if self.has_meaningful_content(reply.text):
                                # Generate and send response
                                response = self.generate_response(
                                    prompt=f"Respond to reply: {reply.text}",
                                    context=context
                                )
                                
                                if response:
                                    # Attempt to post reply
                                    result = self._make_twitter_request(lambda: self.Client.create_tweet(
                                        text=response[:280],
                                        in_reply_to_tweet_id=reply.id,
                                        user_auth=True
                                    ))
                                    
                                    if result and result.data:
                                        # Record the successful reply
                                        self.rate_limiter.record_action("reply")
                                        
                                        # Store reply and interaction data
                                        self.tweet_cache.add_reply(reply.id, reply.text, response)
                                        self.learning_store.store_interaction(
                                            text=reply.text,
                                            response=response,
                                            conversation_id=reply.conversation_id  
                                        )
                                        
                                        logging.info(f"Successfully replied to tweet {reply.id}")
                                        
                                        # Mark reply as processed
                                        self.tweet_cache.mark_tweet_processed(reply.id)
                                    else:
                                        logging.error(f"Failed to post reply to tweet {reply.id}")
                                else:
                                    logging.info(f"No response generated for reply {reply.id}")
                            else:
                                logging.info(f"Reply {reply.id} did not meet content criteria for response")
                            
                            # Always mark as processed
                            self.tweet_cache.mark_tweet_processed(reply.id)
                            
                        except Exception as reply_error:
                            logging.error(f"Error processing individual reply {reply.id}: {str(reply_error)}")
                            self.tweet_cache.mark_tweet_processed(reply.id)  # Mark as processed even if error
                            continue
                            
                except Exception as tweet_error:  
                    logging.error(f"Error processing tweet {tweet['id']}: {str(tweet_error)}")
                    continue
                        
        except Exception as e:
            logging.error(f"Error handling tweet replies: {e}")
            logging.exception("Full traceback:")
            
    def _get_reply_context(self, reply):
        """Helper method to get full context for a reply"""
        try:
            context = []
            
            # Get the original tweet this is replying to
            if reply.referenced_tweets:
                for ref in reply.referenced_tweets:
                    if ref.type == 'replied_to':
                        parent = self._make_twitter_request(lambda: self.Client.get_tweet(
                            ref.id,
                            tweet_fields=['text', 'author_id', 'created_at']
                        ))
                        if parent and parent.data:
                            context.append({
                                'type': 'parent',
                                'text': parent.data.text,
                                'author_id': parent.data.author_id
                            })
            
            # Add the current reply
            context.append({
                'type': 'reply',
                'text': reply.text,
                'author_id': reply.author_id
            })
            
            return context
            
        except Exception as e:
            logging.error(f"Error getting reply context: {e}")
            return [{'type': 'reply', 'text': reply.text, 'author_id': reply.author_id}]
        
    def discover_tweets(self, limit=5):
        try:
            timeline = self._make_twitter_request(
                lambda: self.Client.get_home_timeline(
                    max_results=10,
                    tweet_fields=['public_metrics', 'created_at', 'text'],
                    user_auth=True
                )
            )
            
            if not timeline or not timeline.data:
                return []
                
            interesting_tweets = []
            for tweet in timeline.data:
                # Skip if we've already processed this tweet or replied to it
                if (self.tweet_cache.has_processed_tweet(tweet.id) or 
                    self.tweet_cache.has_processed_reply(tweet.id)):
                    continue
                    
                metrics = tweet.public_metrics
                engagement = (metrics['like_count'] * 2 + 
                            metrics['retweet_count'] * 3 + 
                            metrics['reply_count'])
                if engagement >= 5:
                    interesting_tweets.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'author_id': tweet.author_id,
                        'engagement': engagement,
                        'created_at': tweet.created_at
                    })
            
            interesting_tweets.sort(key=lambda x: x['engagement'], reverse=True)
            return interesting_tweets[:limit]
                
        except Exception as e:
            logging.error(f"Error discovering tweets: {e}")
            return []

    def engage_with_tweet(self, tweet):
        """Engage with a discovered tweet"""
        try:
            # Check if we should engage
            if not self.should_engage(tweet):
                return False
                
            # Generate a response using the tweet context
            response = self.generate_response(
                prompt=f"Respond to this tweet: {tweet['text']} with the following response style: {self.persona['response_style']}",
                context=f"Engaging with tweet about: {tweet['text']}"
            )
            
            if not response:
                return False
                
            # Post the reply
            result = self._make_twitter_request(
                lambda: self.Client.create_tweet(
                    text=response[:280],
                    in_reply_to_tweet_id=tweet['id']
                )
            )
            
            if result and result.data:
                # Record the successful engagement
                self.tweet_cache.add_reply(tweet['id'], tweet['text'], response)
                self.rate_limiter.record_action("reply")
                
                # Store interaction data
                self.learning_store.store_interaction(
                    text=tweet['text'],
                    response=response,
                    engagement=tweet['engagement']
                )
                
                logging.info(f"Successfully engaged with tweet {tweet['id']}")
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error engaging with tweet: {e}")
            return False

    def should_engage(self, tweet):
        """Enhanced engagement criteria for following"""
        # Check rate limits first
        can_reply, _ = self.rate_limiter.check_limit("reply")
        if not can_reply:
            return False
            
        try:
            # Get me for verification
            me = self._make_twitter_request(lambda: self.Client.get_me())
            if not me or not me.data:
                return False
                
            # Don't engage with tweets that:
            # 1. Have too many or too few engagements
            if tweet['engagement'] < 2 or tweet['engagement'] > 1000:  # Lowered minimum engagement for following
                return False
                
            # 2. Contain potentially controversial terms
            controversial_terms = ['politics', 'religion', 'nsfw']
            if any(term in tweet['text'].lower() for term in controversial_terms):
                return False
                
            # 3. Are too short to generate meaningful response
            if len(tweet['text'].split()) < 5:
                return False
                
            # Run content analysis focusing on meaningful conversations
            analysis_prompt = f"""
            Analyze if this tweet from someone we follow is suitable for engagement (YES/NO):
            Tweet: {tweet['text']}
            
            Criteria:
            1. Contains interesting discussion points
            2. Not controversial or inflammatory
            3. Opportunity for meaningful response
            4. Not promotional or spam content
            5. Could lead to valuable conversation
            """
            
            response = self.model.generate_content(
                analysis_prompt,
                generation_config={'temperature': 0.1}
            )
            
            return 'YES' in response.text.upper()
            
        except Exception as e:
            logging.error(f"Error in should_engage: {e}")
            return False
        
    def check_health(self):
        """Verify bot health and configuration"""
        try:
            # Check API connectivity
            me = self._make_twitter_request(lambda: self.Client.get_me())
            if not me or not me.data:
                return False, "Failed to connect to Twitter API"
            
            # Check rate limit status
            status = self.get_rate_limit_status()
            if not status:
                return False, "Failed to get rate limit status"
            
            if not self.learning_store.check_connection():
                return False, "Failed to connect to learning store"
            
            return True, "Bot is healthy"
            
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return False, str(e)
        
    def reset_state(self):
        """Reset bot state - useful for recovery"""
        self._last_tweet_time = None
        self._last_mention_check = None
        self._last_discovery = None
        logging.info("Bot state reset")
    
@app.post("/tweet")
async def create_tweet():
    logging.info("Tweet endpoint called")  # Add logging
    try:
        bot = GeminiTwitterBot()
        logging.info("Bot initialized")
        
        # Use a direct tweet with default content
        response = bot.tweet()  # Simplified call
        
        if response:
            logging.info(f"Tweet successful: {response.data['id']}")
            return {
                "status": "success", 
                "tweet_id": response.data['id'],
                "tweet_url": f"https://twitter.com/status/{response.data['id']}"
            }
        else:
            logging.error("Tweet failed - no response")
            return {"status": "failed"}
            
    except Exception as e:
        logging.error(f"Tweet error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/test-auth")
async def test_auth():
    try:
        bot = GeminiTwitterBot()
        me = bot.Client.get_me()
        if me and me.data:
            return {
                "status": "success",
                "user": {
                    "id": me.data.id,
                    "username": me.data.username
                }
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/check-tweets")
async def check_tweets():
    try:
        bot = GeminiTwitterBot()
        
        # Add delay between API calls
        time.sleep(2)  # 2 second delay
        me = bot.Client.get_me()
        
        # Add delay before getting tweets
        time.sleep(2)
        recent_tweets = bot.get_recent_tweets(limit=5)  # Reduced limit to 5
        
        return {
            "status": "success",
            "user_info": {
                "username": me.data.username,
                "id": me.data.id
            },
            "timeline_check": {
                "has_tweets": len(recent_tweets) > 0,
                "recent_tweets": recent_tweets
            },
            "auth_status": "valid"
        }
    except tweepy.errors.TooManyRequests:
        logging.warning("Rate limit reached, waiting 15 minutes")
        time.sleep(900)  # Wait 15 minutes
        return {"status": "error", "message": "Rate limit reached. Please try again later."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/cache/tweets")
async def get_cached_tweets():
    try:
        bot = GeminiTwitterBot()
        tweets = bot.tweet_cache.get_recent_tweets(limit=10)
        return {
            "status": "success",
            "count": len(tweets),
            "tweets": [
                {
                    "id": t[0],
                    "text": t[1],
                    "url": t[2]
                } for t in tweets
            ]
        }
    except Exception as e:
        logging.error(f"Error retrieving cached tweets: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/learning/interactions")
async def get_learning_interactions():
    try:
        bot = GeminiTwitterBot()
        interactions = bot.learning_store.get_recent_interactions()
        return {
            "status": "success",
            "count": len(interactions),
            "interactions": interactions
        }
    except Exception as e:
        logging.error(f"Error retrieving learning interactions: {e}")
        return {"status": "error", "message": str(e)}
    
@app.post("/cache/clear")
async def clear_cache():
    try:
        bot = GeminiTwitterBot()
        bot.tweet_cache.clear_cache()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logging.error(f"Error clearing cache: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/rate-limits")
async def check_rate_limits():
    try:
        bot = GeminiTwitterBot()
        status = bot.get_rate_limit_status()
        
        # Get last tweet info
        recent_tweets = bot.tweet_cache.get_recent_tweets(limit=1)
        if recent_tweets:
            last_tweet_time = recent_tweets[0][3]
            hours_since = (datetime.now(timezone.utc) - last_tweet_time).total_seconds() / 3600
            status['hours_since_last_tweet'] = round(hours_since, 2)
        
        return {
            "status": "success",
            "rate_limits": status
        }
    except Exception as e:
        logging.error(f"Error checking rate limits: {e}")
        return {"status": "error", "message": str(e)}
    
@app.get("/status")
async def get_bot_status():
    try:
        bot = GeminiTwitterBot()
        rate_limits = bot.get_rate_limit_status()
        recent_tweets = bot.get_recent_tweets(limit=1)
        
        return {
            "status": "running",
            "configured": bot.is_configured,
            "last_tweet": recent_tweets[0] if recent_tweets else None,
            "rate_limits": rate_limits,
            "environment": bot.environment
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    try:
        os.environ['ENVIRONMENT'] = 'prod'
        logging.info("Starting bot in production mode")
        
        # Load environment variables
        load_dotenv()
        
        env_vars = {
            'TWITTER_API_KEY': bool(os.getenv('TWITTER_API_KEY')),
            'TWITTER_API_SECRET': bool(os.getenv('TWITTER_API_SECRET')),
            'TWITTER_ACCESS_TOKEN': bool(os.getenv('TWITTER_ACCESS_TOKEN')),
            'TWITTER_ACCESS_TOKEN_SECRET': bool(os.getenv('TWITTER_ACCESS_TOKEN_SECRET')),
            'TWITTER_BEARER_TOKEN': bool(os.getenv('TWITTER_BEARER_TOKEN'))
        }
        logging.info(f"Environment variables loaded: {env_vars}")
        
        # Initialize bot with retry mechanism
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                bot = GeminiTwitterBot()
                asyncio.create_task(run_bot(bot))
                logging.info("Bot background task created successfully")
                break
            except Exception as e:
                retry_count += 1
                logging.error(f"Bot initialization attempt {retry_count} failed: {e}")
                if retry_count >= max_retries:
                    logging.critical("Failed to initialize bot after maximum retries")
                    raise
                await asyncio.sleep(30)  # Wait before retry
                
    except Exception as e:
        logging.error(f"Failed to start bot: {e}")
        logging.exception("Full traceback:")

async def run_bot_with_retry():
    while True:
        try:
            bot = GeminiTwitterBot()
            await run_bot(bot)
            break
        except Exception as e:
            logging.error(f"Bot initialization failed, retrying in 60 seconds: {e}")
            await asyncio.sleep(60)
        
@app.on_event("shutdown")
async def shutdown_event():
    # Allow time for graceful shutdown
    await asyncio.sleep(2)
    logging.info("Shutting down bot...")

async def run_bot(bot):
    await asyncio.sleep(5)  # Add a short delay to allow initialization
    logging.info("Bot started running")
    
    error_count = 0
    max_errors = 5
    
    while True:
        try:
            current_time = datetime.now(timezone.utc)
            tasks = []
            
            # 1. Handle Tweeting - reduced intervals
            if can_tweet := bot.rate_limiter.check_limit("tweet")[0]:
                if bot.should_tweet():
                    tasks.append(asyncio.create_task(
                        asyncio.to_thread(bot.tweet)
                    ))
                    await asyncio.sleep(5)  # Small delay between operations
            
            # 2. Handle Mentions - more frequent checks
            if can_mention := bot.rate_limiter.check_limit("mention")[0]:
                if (not bot._last_mention_check or 
                    (current_time - bot._last_mention_check).total_seconds() >= 300):  # Check every 5 minutes
                    last_mention = bot.tweet_cache.get_recent_tweets(limit=1)
                    since_id = last_mention[0][0] if last_mention else None
                    tasks.append(asyncio.create_task(
                        asyncio.to_thread(bot.handle_mentions, since_id)
                    ))
                    bot._last_mention_check = current_time
                    await asyncio.sleep(5)
            
            # 3. Handle Discovery and Replies - more frequent
            if can_reply := bot.rate_limiter.check_limit("reply")[0]:
                if (not bot._last_discovery or 
                    (current_time - bot._last_discovery).total_seconds() >= 600):  # Every 10 minutes
                    tasks.append(asyncio.create_task(
                        handle_discovery_and_replies(bot)
                    ))
                    bot._last_discovery = current_time
                    await asyncio.sleep(5)
            
            # Run tasks concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                error_count = 0  # Reset error count on successful execution
            
            # Adaptive sleep based on activity
            if any([can_tweet, can_mention, can_reply]):
                await asyncio.sleep(30)  # Short sleep if active
            else:
                await asyncio.sleep(60)  # Longer sleep if rate limited
                
        except Exception as e:
            error_count += 1
            logging.error(f"Error in main loop: {e}")
            
            if error_count >= max_errors:
                logging.critical("Too many errors, restarting bot...")
                await restart_bot(bot)
                error_count = 0
            
            await asyncio.sleep(60)  # Error cooldown

async def handle_discovery_and_replies(bot):
    """Separate handler for discovery and replies"""
    try:
        discovered_tweets = await asyncio.to_thread(bot.discover_tweets)
        if discovered_tweets:
            for tweet in discovered_tweets:
                await asyncio.to_thread(bot.engage_with_tweet, tweet)
                await asyncio.sleep(2)  # Small delay between engagements
            
            await asyncio.to_thread(bot.handle_tweet_replies)
            
        return True
    except Exception as e:
        logging.error(f"Error in discovery/replies: {e}")
        return False

async def restart_bot(bot):
    try:
        # Reset critical components
        bot.reset_state()
        bot.rate_limiter = RateLimiter()
        
        # Verify API connections
        me = await asyncio.to_thread(lambda: bot.Client.get_me())
        if not me or not me.data:
            raise ValueError("Failed to authenticate after restart")
            
        logging.info("Bot successfully restarted")
        
    except Exception as e:
        logging.error(f"Error during restart: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    os.environ['ENVIRONMENT'] = 'prod'
    bot = GeminiTwitterBot()
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_bot(bot))
    except KeyboardInterrupt:
        logging.info("Bot shutdown requested")
    finally:
        loop.close()