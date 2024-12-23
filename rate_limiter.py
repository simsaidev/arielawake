import os
import logging
from datetime import datetime, timedelta, timezone
from collections import deque

class RateLimiter:
    def __init__(self):
        # Initialize rate limit settings
        self.tweet_timestamps = deque(maxlen=int(os.getenv('BOT_MAX_TWEETS_PER_HOUR', 1)))
        self.reply_timestamps = deque(maxlen=int(os.getenv('BOT_MAX_RESPONSES_PER_HOUR', 3)))
        self.mention_timestamps = deque(maxlen=int(os.getenv('BOT_MAX_MENTIONS_PER_HOUR', 3)))
        
        # Minimum intervals (in seconds)
        self.min_intervals = {
            "tweet": int(os.getenv('BOT_MIN_TWEET_INTERVAL', 3600)),  # Default 1 hour
            "reply": int(os.getenv('BOT_MIN_REPLY_INTERVAL', 900)),   # Default 15 minutes
            "mention": int(os.getenv('BOT_MIN_MENTION_INTERVAL', 900)) # Default 15 minutes
        }
        
        # Track rate limit states
        self.rate_limit_states = {
            "tweet": {"reset_time": None, "is_limited": False, "window_start": None, "last_action": None},
            "reply": {"reset_time": None, "is_limited": False, "window_start": None, "last_action": None},
            "mention": {"reset_time": None, "is_limited": False, "window_start": None, "last_action": None}
        }

    def check_limit(self, action_type="tweet"):
        current_time = datetime.now(timezone.utc)
        state = self.rate_limit_states[action_type]
        
        logging.debug(f"Checking rate limit for {action_type} at {current_time.isoformat()}")
        
        # Check rate-limited state
        if state["is_limited"] and state["reset_time"]:
            if current_time < state["reset_time"]:
                wait_time = (state["reset_time"] - current_time).total_seconds()
                logging.warning(f"Rate limited for {action_type}. Waiting for {wait_time:.0f} seconds.")
                return False, f"Rate limited. Wait {wait_time:.0f} seconds."
        
        # Check minimum interval
        if state["last_action"]:
            time_since_last = (current_time - state["last_action"]).total_seconds()
            if time_since_last < self.min_intervals[action_type]:
                wait_time = self.min_intervals[action_type] - time_since_last
                logging.warning(f"Minimum interval not met for {action_type}. Waiting for {wait_time:.0f} seconds.")
                return False, f"Too soon. Wait {wait_time:.0f} seconds."
        
        # Check hourly limits
        if not self._check_hourly_limit(action_type, current_time):
            logging.warning(f"Hourly limit reached for {action_type}.")
            return False, f"Hourly limit reached for {action_type}."
        
        logging.debug(f"Rate limit check passed for {action_type}.")
        return True, "Action allowed"

    def record_action(self, action_type="tweet"):
        """Record an action to enforce rate limits."""
        current_time = datetime.now(timezone.utc)
        timestamps = getattr(self, f"{action_type}_timestamps", None)
        
        if timestamps is not None:
            timestamps.append(current_time)
        
        # Update state
        self.rate_limit_states[action_type]["last_action"] = current_time
        logging.info(f"Recorded {action_type} action at {current_time.isoformat()}")

    def get_status(self):
        """Retrieve the current rate limit status."""
        current_time = datetime.now(timezone.utc)
        status = {}
        
        for action_type in ["tweet", "reply", "mention"]:
            timestamps = getattr(self, f"{action_type}_timestamps", deque())
            active_timestamps = [ts for ts in timestamps if current_time - ts < timedelta(hours=1)]
            
            rate_limit_info = self.rate_limit_states[action_type]
            
            status[action_type] = {
                "used_this_hour": len(active_timestamps),
                "max_per_hour": timestamps.maxlen,
                "remaining": timestamps.maxlen - len(active_timestamps),
                "is_limited": rate_limit_info["is_limited"],
                "rate_limit_reset": rate_limit_info["reset_time"].isoformat() if rate_limit_info["reset_time"] else None,
            }
        
        return status

    def handle_twitter_rate_limit(self, action_type, reset_time):
        """Handle Twitter API rate limit responses."""
        try:
            reset_datetime = datetime.fromtimestamp(reset_time, timezone.utc)
            self.rate_limit_states[action_type].update({
                "is_limited": True,
                "reset_time": reset_datetime,
                "window_start": datetime.now(timezone.utc)
            })
            
            # Clear timestamps for this action type
            getattr(self, f"{action_type}_timestamps", deque()).clear()
            
            logging.warning(f"Twitter rate limit hit for {action_type}. Reset at {reset_datetime}")
        except Exception as e:
            logging.error(f"Error handling rate limit: {e}")

    def _check_hourly_limit(self, action_type, current_time):
        """Check hourly limits for a given action type."""
        timestamps = getattr(self, f"{action_type}_timestamps", None)
        if timestamps is None:
            return False
        
        active_timestamps = deque(
            [ts for ts in timestamps if (current_time - ts) < timedelta(hours=1)],
            maxlen=timestamps.maxlen
        )
        
        # Update timestamps queue
        setattr(self, f"{action_type}_timestamps", active_timestamps)
        
        return len(active_timestamps) < timestamps.maxlen
