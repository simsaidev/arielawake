import psycopg2
from datetime import datetime, timezone
import logging
import os

class TweetCache:
    def __init__(self, environment=None):
        self.environment = environment or os.getenv('ENVIRONMENT', 'dev')
        self.db_url = os.getenv('DATABASE_URL')
        
        if self.environment == 'dev':
            logging.info("Development environment detected, skipping database initialization")
            return
        
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self._init_db()

    def _init_db(self):
        """Initialize the tweets cache and mention history tables"""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                # Create tables if they don't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tweet_cache (
                        id BIGINT PRIMARY KEY,
                        text TEXT NOT NULL,
                        url TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        environment VARCHAR(10)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_tweet_cache_env_time 
                    ON tweet_cache(environment, timestamp);
                    
                    CREATE TABLE IF NOT EXISTS mention_history (
                        id BIGINT PRIMARY KEY,
                        text TEXT NOT NULL,
                        response TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        processed BOOLEAN DEFAULT FALSE
                    );

                    CREATE TABLE IF NOT EXISTS reply_history (
                        id BIGINT PRIMARY KEY,
                        text TEXT NOT NULL, 
                        response TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        processed BOOLEAN DEFAULT FALSE
                    );
                    
                    CREATE TABLE IF NOT EXISTS processed_tweets (
                        id BIGINT PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)

                # Add processed column if it doesn't exist (for existing tables)
                cur.execute("""
                    DO $$ 
                    BEGIN
                        BEGIN
                            ALTER TABLE mention_history 
                            ADD COLUMN processed BOOLEAN DEFAULT FALSE;
                        EXCEPTION 
                            WHEN duplicate_column THEN 
                            NULL;
                        END;
                        
                        BEGIN
                            ALTER TABLE reply_history 
                            ADD COLUMN processed BOOLEAN DEFAULT FALSE;
                        EXCEPTION 
                            WHEN duplicate_column THEN 
                            NULL;
                        END;
                    END $$;
                """)
                conn.commit()
                
    def add_tweet(self, tweet_id, text, url=None):
        """Add a tweet to the cache"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO tweet_cache (id, text, url, timestamp, environment)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (tweet_id, text, url, datetime.now(timezone.utc), self.environment))
                    conn.commit()
                    logging.info(f"Added tweet to cache: {tweet_id}")
        except Exception as e:
            logging.error(f"Error adding tweet to cache: {e}")

    def add_mention(self, mention_id, text, response):
        """Add a mention to the history"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO mention_history (id, text, response, timestamp)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (mention_id, text, response, datetime.now(timezone.utc)))
                    conn.commit()
                    logging.info(f"Added mention to history: {mention_id}")
        except Exception as e:
            logging.error(f"Error adding mention to history: {e}")

    def has_responded(self, mention_id):
        """Check if a mention has been responded to"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 1 FROM mention_history WHERE id = %s
                    """, (mention_id,))
                    return cur.fetchone() is not None
        except Exception as e:
            logging.error(f"Error checking mention history: {e}")
            return False

    def get_recent_tweets(self, limit=100):
        """Retrieve recent tweets from the cache"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, text, url, timestamp
                        FROM tweet_cache
                        WHERE environment = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (self.environment, limit))
                    
                    tweets = cur.fetchall()
                    return tweets  # Returns (id, text, url, timestamp)
        except Exception as e:
            logging.error(f"Error getting recent tweets: {e}")
            return []

    def clear_cache(self):
        """Clear the tweet cache for the current environment"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM tweet_cache
                        WHERE environment = %s
                    """, (self.environment,))
                    conn.commit()
                    logging.info("Cache cleared")
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")

    def cleanup_old_tweets(self, days=30):
        """Remove tweets older than specified days"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM tweet_cache
                        WHERE environment = %s
                        AND timestamp < NOW() - INTERVAL '%s days'
                    """, (self.environment, days))
                    conn.commit()
                    logging.info(f"Cleaned up tweets older than {days} days")
        except Exception as e:
            logging.error(f"Error cleaning up old tweets: {e}")
            
    def has_processed_mention(self, mention_id):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT processed FROM mention_history WHERE id = %s
                    """, (mention_id,))
                    result = cur.fetchone()
                    return bool(result and result[0])
        except Exception as e:
            logging.error(f"Error checking processed mention: {e}")
            return False

    def has_processed_reply(self, reply_id):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT processed FROM reply_history WHERE id = %s
                    """, (reply_id,))
                    result = cur.fetchone()
                    return bool(result and result[0])
        except Exception as e:
            logging.error(f"Error checking processed reply: {e}")
            return False

    def has_processed_tweet(self, tweet_id):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT EXISTS(SELECT 1 FROM processed_tweets WHERE id = %s)", (str(tweet_id),))
                    return cur.fetchone()[0]
        except Exception as e:
            logging.error(f"Error checking processed tweet: {e}")
            return False
    
    def mark_tweet_processed(self, tweet_id):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO processed_tweets (id) VALUES (%s) ON CONFLICT (id) DO NOTHING", (str(tweet_id),))
                    conn.commit()
        except Exception as e:
            logging.error(f"Error marking tweet as processed: {e}")
            
    def mark_mention_processed(self, mention_id):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO mention_history (id, text, processed)
                        VALUES (%s, 'processed', TRUE)
                        ON CONFLICT (id) DO UPDATE SET processed = TRUE
                    """, (mention_id,))
                    conn.commit()
        except Exception as e:
            logging.error(f"Error marking mention processed: {e}")

    def mark_reply_processed(self, reply_id):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO reply_history (id, text, processed)
                        VALUES (%s, 'processed', TRUE)
                        ON CONFLICT (id) DO UPDATE SET processed = TRUE
                    """, (reply_id,))
                    conn.commit()
        except Exception as e:
            logging.error(f"Error marking reply processed: {e}")
            
    def load_recent_timestamps(self):
        """Load timestamps of recent tweets for rate limiting"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Get tweets from last hour for rate limiting
                    cur.execute("""
                        SELECT timestamp 
                        FROM tweet_cache 
                        WHERE environment = %s
                        AND timestamp > NOW() - INTERVAL '1 hour'
                        ORDER BY timestamp DESC
                    """, (self.environment,))
                    
                    # Return list of timestamps
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logging.error(f"Error loading recent timestamps: {e}")
            return []
        
    def check_connection(self):
        """Verify database connection status"""
        if self.environment == 'dev':
            return True
            
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return cur.fetchone() is not None
        except Exception as e:
            logging.error(f"Database connection check failed: {e}")
            return False

    def add_reply(self, reply_id, text, response):
        """Add a reply to the history"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO reply_history (id, text, response, timestamp)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (reply_id, text, response, datetime.now(timezone.utc)))
                    conn.commit()
                    logging.info(f"Added reply to history: {reply_id}")
        except Exception as e:
            logging.error(f"Error adding reply to history: {e}")
            
    def cleanup_old_history(self, days=30):
        """Clean up old mentions and replies"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM mention_history 
                        WHERE timestamp < NOW() - INTERVAL '%s days';
                        DELETE FROM reply_history 
                        WHERE timestamp < NOW() - INTERVAL '%s days';
                    """, (days, days))
                    conn.commit()
                    logging.info("Cleaned up old mention and reply history")
        except Exception as e:
            logging.error(f"Error cleaning history: {e}")
            
    def get_cache_stats(self):
        """Get cache sta`tistics"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_tweets,
                            COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') as last_hour,
                            MIN(timestamp) as oldest_tweet,
                            MAX(timestamp) as newest_tweet
                        FROM tweet_cache
                        WHERE environment = %s
                    """, (self.environment,))
                    return cur.fetchone()
        except Exception as e:
            logging.error(f"Error getting cache stats: {e}")
            return None