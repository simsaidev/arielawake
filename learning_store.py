# learning_store.py
import os
import psycopg2
from datetime import datetime, timezone, timedelta
import logging

class LearningStore:
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'dev')
        self.db_url = os.getenv('DATABASE_URL', '').strip()
        
        if self.environment == 'dev':
            logging.info("Development environment detected, skipping database initialization")
            return
        
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self._init_db()
        
    def _init_db(self):
        """Create necessary tables if they don't exist"""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                # Create tables with updated schema
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        id SERIAL PRIMARY KEY,
                        text TEXT,
                        response TEXT,
                        engagement INTEGER DEFAULT 0,
                        uniqueness_score INTEGER DEFAULT 0,
                        creativity_notes TEXT,
                        evolution_analysis TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        url TEXT,
                        conversation_id TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS evolution_patterns (
                        id SERIAL PRIMARY KEY,
                        text TEXT,
                        engagement INTEGER,
                        uniqueness INTEGER,
                        insights TEXT,
                        evolution TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    -- Add conversation_id column if it doesn't exist
                    DO $$ 
                    BEGIN
                        BEGIN
                            ALTER TABLE interactions 
                            ADD COLUMN conversation_id TEXT;
                        EXCEPTION 
                            WHEN duplicate_column THEN 
                            NULL;
                        END;
                    END $$;
                """)
                conn.commit()

    def store_interaction(self, text, response=None, engagement=0, uniqueness_score=0, 
                     creativity_notes=None, evolution_analysis=None, timestamp=None, 
                     conversation_id=None):
        """Store interaction in PostgreSQL with proper null handling"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO interactions 
                        (text, response, engagement, uniqueness_score, creativity_notes, 
                        evolution_analysis, timestamp, conversation_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (text, response, engagement, uniqueness_score, creativity_notes, 
                        evolution_analysis, timestamp, conversation_id))
                    
                    interaction_id = cur.fetchone()[0]
                    
                    if engagement > 5 or uniqueness_score > 7:
                        cur.execute("""
                            INSERT INTO evolution_patterns 
                            (text, engagement, uniqueness, insights, evolution, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (text, engagement, uniqueness_score, creativity_notes, evolution_analysis, timestamp))
                    
                    conn.commit()
                    logging.info(f"Stored interaction {interaction_id} with engagement {engagement}")
                    return True
        except Exception as e:
            logging.error(f"Error storing interaction: {e}")
            logging.exception("Full error:")  # Log full stack trace for debugging
            return False
        
    def get_successful_patterns(self, limit=10):
        """Get patterns that showed high engagement or creativity"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        (SELECT text, engagement, uniqueness_score as uniqueness, 
                                creativity_notes as insights, evolution_analysis as evolution, 
                                timestamp
                         FROM interactions 
                         WHERE engagement > 5 OR uniqueness_score > 7)
                        UNION ALL
                        (SELECT text, engagement, uniqueness, 
                                insights, evolution, timestamp 
                         FROM evolution_patterns)
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                    
                    columns = ['text', 'engagement', 'uniqueness', 'insights', 'evolution', 'timestamp']
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            logging.error(f"Error getting patterns: {e}")
            return []

    def get_recent_interactions(self, limit=10):
        """Get recent interactions"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT text, response, engagement, uniqueness_score, 
                               creativity_notes, evolution_analysis, timestamp
                        FROM interactions
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                    
                    columns = ['text', 'response', 'engagement', 'uniqueness_score', 
                             'creativity_notes', 'evolution_analysis', 'timestamp']
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            logging.error(f"Error getting recent interactions: {e}")
            return []
    
    def cleanup_old_data(self):
        """Remove data older than 30 days"""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Clean old interactions
                    cur.execute("DELETE FROM interactions WHERE timestamp < %s", (cutoff,))
                    # Clean old evolution patterns
                    cur.execute("DELETE FROM evolution_patterns WHERE timestamp < %s", (cutoff,))
                    conn.commit()
                    
                    logging.info("Cleaned up old data successfully")
        except Exception as e:
            logging.error(f"Error cleaning up old data: {e}")

    def check_database(self):
        """Check database status and contents"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM interactions")
                    interactions_count = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM evolution_patterns")
                    patterns_count = cur.fetchone()[0]
                    
                    cur.execute("""
                        SELECT text, engagement, timestamp 
                        FROM interactions 
                        ORDER BY timestamp DESC 
                        LIMIT 5
                    """)
                    recent = cur.fetchall()
                    
                    return {
                        'interactions_count': interactions_count,
                        'patterns_count': patterns_count,
                        'recent_interactions': recent
                    }
        except Exception as e:
            logging.error(f"Database check error: {e}")
            return None
        
    def check_connection(self):
        """Verify database connection is working"""
        if self.environment == 'dev':
            return True
            
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Simple query to test connection
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    return result is not None
        except Exception as e:
            logging.error(f"Database connection check failed: {e}")
            return False
        
    def reset(self):
        """Reset the learning store state"""
        if self.environment != 'dev':
            try:
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        # Truncate tables but keep structure
                        cur.execute("TRUNCATE interactions, evolution_patterns")
                        conn.commit()
                        logging.info("Learning store reset successfully")
            except Exception as e:
                logging.error(f"Error resetting learning store: {e}")
    
    def get_engagement_stats(self):
        """Get engagement statistics for analysis"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            AVG(engagement) as avg_engagement,
                            MAX(engagement) as max_engagement,
                            AVG(uniqueness_score) as avg_uniqueness
                        FROM interactions
                        WHERE timestamp > NOW() - INTERVAL '24 hours'
                    """)
                    return cur.fetchone()
        except Exception as e:
            logging.error(f"Error getting engagement stats: {e}")
            return None