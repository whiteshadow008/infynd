from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from textblob import TextBlob
import random
from datetime import datetime, timedelta
import re
from collections import defaultdict
from typing import List, Dict, Any
from dataclasses import dataclass, field
import json
import traceback

# ===========================
# CONFIGURATION & MOCK SETUP
# ===========================

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@dataclass
class Post:
    """Structured data class for a single social media mention."""
    post_id: str
    text: str
    platform: str
    brand: str
    timestamp: str
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    sentiment: str = 'Neutral'
    polarity: float = 0.0
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    emotions: Dict[str, int] = field(default_factory=dict)
    location: str = 'Unknown'
    language: str = 'en'
    user_segment: str = 'General'

class MockDatabase:
    """
    Simulates a persistent database (e.g., MongoDB/PostgreSQL) for storing
    all collected social media posts and tracking active streams.
    """
    def __init__(self):
        # Stores all collected historical posts keyed by brand name
        self.posts_by_brand: Dict[str, List[Post]] = defaultdict(list)
        # Stores active streams for real-time monitoring
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        # Stores alert thresholds
        self.alert_thresholds: Dict[str, Dict[str, Any]] = {}

    def insert_post(self, post: Post):
        """Inserts a post into the database."""
        self.posts_by_brand[post.brand].append(post)

    def get_posts(self, brand: str, time_period: str = '7d') -> List[Post]:
        """Simulates querying historical data for a brand."""
        # For simulation, this will trigger sample data generation if not found
        if not self.posts_by_brand[brand]:
            # Simulate a "deep dive" historical fetch for this brand
            self.posts_by_brand[brand] = self._generate_varied_data(brand, time_period)
        
        # Filter posts by time period for realism (though data is static)
        cutoff = self._get_cutoff_date(time_period)
        return [p for p in self.posts_by_brand[brand] if datetime.fromisoformat(p.timestamp) >= cutoff]

    def _generate_varied_data(self, brand: str, time_period: str) -> List[Post]:
        """Generates realistic, but varied, sample data for comparison."""
        base_posts = generate_sample_data(num_posts=150)
        posts: List[Post] = []
        
        # Assign brand-specific sentiment profile
        profile = self._get_brand_profile(brand)
        
        for i, post in enumerate(base_posts):
            # Introduce brand-specific bias
            if random.random() < profile['sentiment_bias']:
                post_text = random.choice(profile['keywords'])
                sentiment, polarity = analyze_sentiment_textblob(post_text)
                
                # Force sentiment to align with profile for a subset of posts
                if random.random() < 0.3: # 30% chance to overwrite sentiment
                    sentiment = random.choices(['Positive', 'Negative'], weights=[profile['positive_skew'], 1 - profile['positive_skew']])[0]
                    polarity = random.uniform(0.5, 0.9) if sentiment == 'Positive' else random.uniform(-0.9, -0.5)

            else:
                 sentiment, polarity = analyze_sentiment_textblob(post.get('text', ''))

            # Update base post with new data
            post_id = f"{brand}_{i}_{datetime.now().timestamp()}"
            hashtags, mentions, keywords = extract_keywords(post.get('text', ''))
            
            p = Post(
                post_id=post_id,
                text=post.get('text', ''),
                platform=random.choice(profile['platforms']),
                brand=brand,
                timestamp=post.get('timestamp', datetime.now().isoformat()),
                likes=post.get('likes', 0),
                replies=post.get('replies', 0),
                sentiment=sentiment,
                polarity=polarity,
                hashtags=hashtags,
                keywords=keywords,
                location=random.choice(profile['locations']),
                user_segment=random.choice(profile['segments'])
            )
            posts.append(p)
            
        return posts

    def _get_brand_profile(self, brand: str) -> Dict[str, Any]:
        """Defines a simulated profile for different brands for realistic comparison."""
        brand_profiles = {
            "Tesla": {
                "sentiment_bias": 0.5, "positive_skew": 0.65,
                "keywords": ["Amazing Tesla EV experience!", "Tesla quality issues", "Love my Model 3!", "Tesla service problems", "Elon Musk innovation"],
                "locations": ["North America", "Europe", "Asia", "Unknown"],
                "platforms": ["Twitter", "Reddit"],
                "segments": ["Tech Enthusiast", "Investor", "General Public"]
            },
            "BMW": {
                "sentiment_bias": 0.3, "positive_skew": 0.55,
                "keywords": ["BMW luxury at its finest", "i4 electric disappointment", "iDrive system rocks", "BMW service too expensive", "M series performance"],
                "locations": ["Europe", "North America", "Unknown"],
                "platforms": ["Facebook", "Instagram", "Twitter"],
                "segments": ["Luxury Buyer", "Auto Enthusiast", "General Public"]
            },
            "Nike": {
                "sentiment_bias": 0.6, "positive_skew": 0.70,
                "keywords": ["Jordan collection fire!", "Nike quality decline", "Air Max comfort amazing", "Nike overpriced lately", "sustainability goals impressive"],
                "locations": ["Asia", "North America", "Europe", "Unknown"],
                "platforms": ["Instagram", "Twitter", "Facebook"],
                "segments": ["Athlete", "Fashion Buyer", "Youth"]
            },
            "Mercedes": {
                "sentiment_bias": 0.2, "positive_skew": 0.50,
                "keywords": ["S-Class ultimate luxury", "Mercedes dealer issues", "EQS electric impressive", "Mercedes quality dropped", "service center problems"],
                "locations": ["Europe", "North America", "South America", "Unknown"],
                "platforms": ["LinkedIn", "Twitter", "Facebook"],
                "segments": ["Luxury Buyer", "Older Demographics", "Investor"]
            }
        }
        
        # Default profile if brand is new
        default_profile = {
            "sentiment_bias": 0.4, "positive_skew": 0.60,
            "keywords": ["Great product quality!", "Disappointed with service", "Amazing experience overall", "Not worth the price", "Highly recommend this!"],
            "locations": ["North America", "Europe", "Unknown"],
            "platforms": ["Twitter", "Reddit", "Facebook"],
            "segments": ["General Public"]
        }
        
        return brand_profiles.get(brand, default_profile)

    def _get_cutoff_date(self, time_period: str) -> datetime:
        """Helper to determine the time cutoff for queries."""
        if time_period == '24h':
            return datetime.now() - timedelta(hours=24)
        elif time_period == '7d':
            return datetime.now() - timedelta(days=7)
        elif time_period == '30d':
            return datetime.now() - timedelta(days=30)
        return datetime.now() - timedelta(days=7)

    def insert_stream_post(self, stream_id: str, post: Post):
        """Inserts a new post into an active stream and the main database."""
        if stream_id in self.active_streams:
            # Add to stream buffer
            self.active_streams[stream_id]['data'].append(post)
            # Check for alerts
            self.check_sentiment_alerts(stream_id, post)
            # Also store in main database for persistence
            self.insert_post(post)
    
    def check_sentiment_alerts(self, stream_id: str, new_post: Post):
        """Enhanced alert checking logic for new posts."""
        if stream_id not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[stream_id]
        alerts = self.active_streams[stream_id].get('alerts', [])
        
        # Check negative sentiment threshold
        if new_post.sentiment == 'Negative' and new_post.polarity < thresholds.get('negative_threshold', -0.5):
            alerts.append({
                'type': 'high_negative_post',
                'severity': 'critical',
                'message': f"Critical negative post detected: {new_post.text[:50]}...",
                'timestamp': datetime.now().isoformat()
            })
            
        # Check for sudden spike in mentions (simulated by checking recent post count)
        recent_posts = self.active_streams[stream_id]['data'][-10:]
        mention_spike_threshold = thresholds.get('mention_spike', 5)
        
        if len(recent_posts) > mention_spike_threshold and len(recent_posts) > 1 and new_post == recent_posts[-1]:
             # Only trigger if the latest post is part of a cluster spike
            alerts.append({
                'type': 'mention_spike',
                'severity': 'warning',
                'message': f"Sudden spike: {len(recent_posts)} posts in a short window.",
                'timestamp': datetime.now().isoformat()
            })
            
        self.active_streams[stream_id]['alerts'] = alerts


# Initialize the global mock database instance
db = MockDatabase()

# ===========================
# SENTIMENT ANALYSIS FUNCTIONS
# ===========================

def analyze_sentiment_textblob(text: str) -> tuple:
    """Sentiment analysis using TextBlob"""
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'Positive', polarity
        elif polarity < -0.1:
            return 'Negative', polarity
        else:
            return 'Neutral', polarity
    except:
        return 'Neutral', 0.0

def detect_emotions(text: str) -> Dict[str, int]:
    """Detect specific emotions in text based on keywords (simulated model)."""
    emotion_keywords = {
        'joy': ['happy', 'love', 'amazing', 'excellent', 'great', 'wonderful', 'excited', 'fantastic'],
        'anger': ['angry', 'hate', 'terrible', 'worst', 'awful', 'horrible', 'furious', 'annoyed'],
        'sadness': ['sad', 'disappointed', 'unhappy', 'depressed', 'unfortunate'],
        'surprise': ['wow', 'amazing', 'unexpected', 'shocked'],
        'fear': ['scared', 'worried', 'concerned', 'nervous', 'afraid', 'anxious']
    }
    
    text_lower = text.lower()
    emotions = {}
    
    for emotion, keywords in emotion_keywords.items():
        count = sum(1 for word in keywords if word in text_lower)
        if count > 0:
            emotions[emotion] = count
    
    return emotions

def extract_keywords(text: str) -> tuple:
    """Extract hashtags, mentions, and keywords."""
    hashtags = re.findall(r'#\w+', str(text))
    mentions = re.findall(r'@\w+', str(text))
    
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
                  'her', 'was', 'one', 'our', 'out', 'this', 'that', 'with', 
                  'have', 'from', 'they', 'been', 'will', 'what', 'more', 'product', 'brand', 'service'}
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    keywords = [w for w in words if w not in stop_words]
    
    return hashtags, mentions, keywords

# ===========================
# DATA GENERATION & REAL-TIME MOCK
# ===========================

def generate_sample_data(num_posts: int = 100) -> List[Dict[str, Any]]:
    """Generate sample social media posts for baseline data."""
    sample_posts = [
        "This product is amazing! Best purchase ever! #love #awesome",
        "Terrible customer service. Very disappointed. #fail",
        "It's okay, nothing special but does the job.",
        "Love the new features! Keep up the great work! @brandname",
        "Worst experience. Will never buy again. #disappointed",
        "Just got the new item, the quality seems questionable.",
        "The price point is surprisingly good for what you get.",
        "Neutral feeling about the update, neither great nor bad.",
        "Outstanding quality and performance! Highly recommend! #excellence",
        "Poor build quality. Expected better for the price. #letdown",
        "Decent product but nothing extraordinary.",
        "Absolutely love this! Worth every penny! #satisfied",
        "Customer support was unhelpful and rude. #frustrated",
        "Works as expected, no complaints so far.",
        "Innovation at its best! Truly impressed! #innovative",
        "Overpriced and underwhelming. Not recommended. #overrated"
    ]
    
    platforms = ['Twitter', 'Facebook', 'Instagram', 'Reddit', 'LinkedIn']
    locations = ['North America', 'Europe', 'Asia', 'South America', 'Unknown']
    user_segments = ["Tech Enthusiast", "Investor", "General Public", "Luxury Buyer", "Youth", "Auto Enthusiast"]
    
    data = []
    for i in range(num_posts):
        post_text = random.choice(sample_posts)
        timestamp = datetime.now() - timedelta(hours=random.randint(0, 72), minutes=random.randint(0, 60))
        
        sentiment, polarity = analyze_sentiment_textblob(post_text)
        hashtags, mentions, keywords = extract_keywords(post_text)
        emotions = detect_emotions(post_text)
        
        data.append({
            'post_id': str(i + 1),
            'text': post_text,
            'platform': random.choice(platforms),
            'timestamp': timestamp.isoformat(),
            'likes': random.randint(0, 500),
            'retweets': random.randint(0, 100),
            'replies': random.randint(0, 50),
            'sentiment': sentiment,
            'polarity': polarity,
            'hashtags': hashtags,
            'mentions': mentions,
            'keywords': keywords,
            'emotions': emotions,
            'location': random.choice(locations),
            'language': 'en',
            'user_segment': random.choice(user_segments)
        })
    
    return data

def start_realtime_stream_mock(brand_name: str, keywords: List[str], platform: str) -> Dict[str, Any]:
    """Mocks starting a real-time stream via API search and initializes stream data in MockDB."""
    try:
        stream_id = f"{brand_name}_{int(datetime.now().timestamp())}_{random.randint(100, 999)}"
        
        # Simulate fetching initial 20 posts from the 'stream'
        mock_posts_data = generate_sample_data(20)
        
        # Process and convert to Post objects
        initial_posts: List[Post] = []
        for i, data in enumerate(mock_posts_data):
            post_id = f"{stream_id}_{i}"
            
            # Introduce very high negative post to test immediate alerts
            if i == 1: 
                data['text'] = f"The {brand_name} launch is an absolute disaster! Shameful. #worstlaunch"
                data['sentiment'], data['polarity'] = 'Negative', -0.95
            
            p = Post(
                post_id=post_id,
                brand=brand_name,
                platform=platform,
                text=data.get('text', ''),
                timestamp=data.get('timestamp', datetime.now().isoformat()),
                likes=data.get('likes', 0),
                retweets=data.get('retweets', 0),
                replies=data.get('replies', 0),
                sentiment=data.get('sentiment', 'Neutral'),
                polarity=data.get('polarity', 0.0),
                hashtags=data.get('hashtags', []),
                mentions=data.get('mentions', []),
                keywords=data.get('keywords', []),
                emotions=data.get('emotions', {}),
                location=data.get('location', 'Unknown'),
                language=data.get('language', 'en'),
                user_segment=data.get('user_segment', 'General')
            )
            initial_posts.append(p)
            db.insert_post(p) # Insert into historical DB
            
        db.active_streams[stream_id] = {
            'brand': brand_name,
            'platform': platform,
            'keywords': keywords,
            'data': initial_posts, # Current buffer of stream posts
            'status': 'active',
            'start_time': datetime.now().isoformat(),
            'alerts': []
        }
        
        # Run alert checks on initial data
        for post in initial_posts:
            db.check_sentiment_alerts(stream_id, post)

        return {'success': True, 'stream_id': stream_id}
    except Exception as e:
        print(f"Error in start_realtime_stream_mock: {str(e)}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

# ===========================
# ANALYSIS FUNCTIONS
# ===========================

def calculate_metrics(data: List[Post]) -> Dict[str, Any]:
    """Calculate comprehensive sentiment metrics."""
    if not data:
        return {'total_posts': 0, 'positive_count': 0, 'negative_count': 0, 'neutral_count': 0,
                'positive_pct': 0, 'negative_pct': 0, 'neutral_pct': 0,
                'avg_polarity': 0.0, 'total_engagement': 0}
    
    df = pd.DataFrame([p.__dict__ for p in data])
    total = len(df)
    
    positive = len(df[df['sentiment'] == 'Positive'])
    negative = len(df[df['sentiment'] == 'Negative'])
    neutral = len(df[df['sentiment'] == 'Neutral'])
    
    return {
        'total_posts': total,
        'positive_count': positive,
        'negative_count': negative,
        'neutral_count': neutral,
        'positive_pct': round((positive / total * 100) if total > 0 else 0, 1),
        'negative_pct': round((negative / total * 100) if total > 0 else 0, 1),
        'neutral_pct': round((neutral / total * 100) if total > 0 else 0, 1),
        'avg_polarity': round(df['polarity'].mean(), 3),
        'total_engagement': int(df['likes'].sum() + df.get('retweets', pd.Series([0])).sum())
    }

def analyze_brand_sentiment(brand_name: str, competitors: List[str], time_period: str) -> Dict[str, Any]:
    """Comprehensive brand sentiment analysis with competitor comparison."""
    all_brands = [brand_name] + competitors
    brand_analysis = {}
    total_mentions = 0
    
    for brand in all_brands:
        # Data is fetched/generated from MockDatabase
        posts = db.get_posts(brand, time_period)
        metrics = calculate_metrics(posts)
        
        brand_analysis[brand] = {
            'data': [p.__dict__ for p in posts],
            'metrics': metrics,
            'sentiment_score': metrics.get('avg_polarity', 0),
            'trend': calculate_sentiment_trend(posts),
            'share_of_voice': 0
        }
        total_mentions += metrics.get('total_posts', 0)
    
    # Calculate share of voice
    for brand in brand_analysis:
        mentions = brand_analysis[brand]['metrics']['total_posts']
        brand_analysis[brand]['share_of_voice'] = (mentions / total_mentions * 100) if total_mentions > 0 else 0
    
    return brand_analysis

def calculate_sentiment_trend(data: List[Post]) -> str:
    """Calculate sentiment trend (improving/declining/stable) based on two halves."""
    if len(data) < 20:
        return 'stable'
    
    df = pd.DataFrame([p.__dict__ for p in data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Split into two halves: oldest (first_half) vs newest (second_half)
    mid = len(df) // 2
    first_half_avg = df.iloc[:mid]['polarity'].mean()
    second_half_avg = df.iloc[mid:]['polarity'].mean()
    
    diff = second_half_avg - first_half_avg
    
    if diff > 0.1:
        return 'improving'
    elif diff < -0.1:
        return 'declining'
    else:
        return 'stable'

def analyze_by_region(data: List[Post]) -> tuple:
    """Analyze sentiment by geographic region and demographic segments."""
    regional_data = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})
    demographics = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})
    
    for post in data:
        region = post.location
        segment = post.user_segment
        sentiment = post.sentiment
        
        # Regional
        regional_data[region][sentiment.lower()] += 1
        regional_data[region]['total'] += 1
        
        # Demographic (by segment type, simulating age/gender for realism)
        demographics[segment][sentiment.lower()] += 1
        demographics[segment]['total'] += 1
    
    # Calculate percentages for Regional
    for region in regional_data:
        total = regional_data[region]['total']
        if total > 0:
            regional_data[region]['positive_pct'] = round(regional_data[region]['positive'] / total * 100, 1)
            regional_data[region]['negative_pct'] = round(regional_data[region]['negative'] / total * 100, 1)
    
    # Calculate percentages for Demographics
    for segment in demographics:
        total = demographics[segment]['total']
        if total > 0:
            demographics[segment]['positive_pct'] = round(demographics[segment]['positive'] / total * 100, 1)
            demographics[segment]['negative_pct'] = round(demographics[segment]['negative'] / total * 100, 1)
    
    return dict(regional_data), dict(demographics)

def analyze_trending_topics(data: List[Post], time_window: str) -> Dict[str, Any]:
    """Analyze trending keywords and hashtags with temporal analysis."""
    df = pd.DataFrame([p.__dict__ for p in data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get recent data based on time window
    cutoff = db._get_cutoff_date(time_window)
    recent_data = df[df['timestamp'] >= cutoff]
    
    # Analyze hashtags with sentiment
    hashtag_analysis = defaultdict(lambda: {'count': 0, 'positive': 0, 'negative': 0, 'neutral': 0})
    
    for _, post in recent_data.iterrows():
        for hashtag in post['hashtags']:
            hashtag_analysis[hashtag]['count'] += 1
            hashtag_analysis[hashtag][post['sentiment'].lower()] += 1
    
    # Calculate sentiment percentages and scores
    for tag in hashtag_analysis:
        total = hashtag_analysis[tag]['count']
        if total > 0:
            hashtag_analysis[tag]['positive_pct'] = round(hashtag_analysis[tag]['positive'] / total * 100, 1)
            hashtag_analysis[tag]['negative_pct'] = round(hashtag_analysis[tag]['negative'] / total * 100, 1)
            hashtag_analysis[tag]['sentiment_score'] = round((hashtag_analysis[tag]['positive'] - hashtag_analysis[tag]['negative']) / total, 3)
        else:
            hashtag_analysis[tag]['positive_pct'] = 0
            hashtag_analysis[tag]['negative_pct'] = 0
            hashtag_analysis[tag]['sentiment_score'] = 0

    # Sort by count and filter to top 20
    trending_hashtags = sorted(hashtag_analysis.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
    
    return {
        'trending_hashtags': [{'tag': tag, **data} for tag, data in trending_hashtags],
        'time_window': time_window,
        'total_analyzed': len(recent_data)
    }

def analyze_keyword_evolution(data: List[Post], keyword: str) -> List[Dict[str, Any]]:
    """Track how sentiment about a specific keyword evolves over time."""
    df = pd.DataFrame([p.__dict__ for p in data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Filter posts containing keyword
    keyword_posts = df[df['text'].str.contains(keyword, case=False, na=False)]
    
    # Group by time periods (daily for simplicity)
    keyword_posts['date'] = keyword_posts['timestamp'].dt.date
    daily_sentiment = keyword_posts.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    
    evolution = []
    for date, row in daily_sentiment.iterrows():
        total = row.sum()
        evolution.append({
            'date': str(date),
            'positive': int(row.get('Positive', 0)),
            'negative': int(row.get('Negative', 0)),
            'neutral': int(row.get('Neutral', 0)),
            'total': int(total),
            'sentiment_score': round((row.get('Positive', 0) - row.get('Negative', 0)) / total if total > 0 else 0, 3)
        })
    
    return evolution

def generate_ai_summary(data: List[Post], brand_name: str) -> str:
    """
    Mocks an interaction with an advanced LLM to generate a sophisticated
    executive summary and recommendations.
    """
    if not data:
        return "No data available for summarization."
    
    # 1. Gather comprehensive inputs
    metrics = calculate_metrics(data)
    regional, demographics = analyze_by_region(data)
    trending = analyze_trending_topics(data, '7d')
    
    # 3. Simulate LLM Output based on context
    trend = calculate_sentiment_trend(data)
    health = "strong and growing" if trend == 'improving' and metrics['avg_polarity'] > 0.1 else "stable but facing minor headwinds"
    
    top_positive_topic = trending['trending_hashtags'][0]['tag'] if trending['trending_hashtags'] else 'Product Quality'
    top_negative_topic = "Customer Service issues"
    
    negative_posts = [p for p in data if p.sentiment == 'Negative']
    sample_negative = negative_posts[0].text[:30] if negative_posts else "service delays"
    
    top_region = list(regional.keys())[0] if regional else "North America"
    second_region = list(regional.keys())[1] if len(regional) > 1 else "Europe"
    top_demo = list(demographics.keys())[0] if demographics else "General Public"
    
    simulated_summary = f"""
🎯 EXECUTIVE SUMMARY — {brand_name} Brand Health

The current brand health index is {health}.  
We observe a net sentiment score of {metrics['avg_polarity']:.3f}, with {metrics['total_posts']} total mentions analyzed.  
The dominant trend is {trend.upper()}.

Key Drivers of Sentiment

• Positive Driver: Conversations around {top_positive_topic} are driving favorable buzz, especially among the {top_demo} demographic.  
• Negative Driver: The topic "{top_negative_topic}" is a recurring friction point. A sample negative post ("{sample_negative}...") highlights customer concerns.

Regional Performance Snapshot

• {top_region} region leads with {regional.get(top_region, {}).get('positive_pct', 0)}% positive sentiment ({regional.get(top_region, {}).get('total', 0)} mentions).  
• {second_region} region shows {regional.get(second_region, {}).get('negative_pct', 0)}% negative sentiment, requiring attention.

Engagement Metrics

• Total Engagement: {metrics['total_engagement']} (likes + shares)
• Sentiment Distribution: {metrics['positive_pct']}% Positive | {metrics['neutral_pct']}% Neutral | {metrics['negative_pct']}% Negative

Actionable Recommendations

1. Amplify positive momentum by promoting user-generated content around {top_positive_topic} in {top_region}.  
2. Address recurring complaints immediately — enforce a 24-hour response SLA for {top_negative_topic} mentions.  
3. Perform deeper analysis by topic cluster, user segment, and platform to surface root causes.  
4. Tailor regional messaging to address concerns in underperforming markets like {second_region}.  
5. Set up real-time alerts for negative sentiment spikes or sudden topic surges.

Strategic Priority: Focus on converting the {metrics['neutral_pct']}% neutral sentiment into positive through targeted engagement campaigns.
"""
    
    return simulated_summary

# ===========================
# FLASK ROUTES
# ===========================

@app.route('/')
def index():
    """Serve the index.html content."""
    try:
        return render_template('sentiment.html')
    except Exception as e:
        print(f"Error serving index: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample data for initial dashboard load."""
    try:
        brand = 'SampleBrand'
        posts = db.get_posts(brand, '7d')
        metrics = calculate_metrics(posts)
        
        return jsonify({
            'success': True,
            'data': [p.__dict__ for p in posts],
            'metrics': metrics
        }), 200
    except Exception as e:
        print(f"Error in get_sample_data: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/brand-analysis', methods=['POST'])
def analyze_brand():
    """Comprehensive brand analysis for a single brand."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        brand_name = data.get('brand_name')
        time_period = data.get('time_period', '7d')
        
        if not brand_name:
            return jsonify({'success': False, 'error': 'Brand name required'}), 400
        
        # Use the unified analysis function, treating it as a single brand analysis
        analysis = analyze_brand_sentiment(brand_name, [], time_period)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        }), 200
    except Exception as e:
        print(f"Error in analyze_brand: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/start-realtime', methods=['POST'])
def start_realtime_monitoring():
    """Start real-time sentiment monitoring mock."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        brand_name = data.get('brand_name')
        keywords = data.get('keywords', [])
        platform = data.get('platform', 'twitter')
        
        if not brand_name:
            return jsonify({'success': False, 'error': 'Brand name required'}), 400
        
        print(f"Starting real-time monitoring for {brand_name} on {platform}")
        result = start_realtime_stream_mock(brand_name, keywords, platform)
        
        if result.get('success'):
            print(f"Real-time stream started successfully: {result.get('stream_id')}")
            return jsonify(result), 200
        else:
            print(f"Failed to start real-time stream: {result.get('error')}")
            # Generate fallback data if stream creation fails
            stream_id = f"{brand_name}_fallback_{int(datetime.now().timestamp())}"
            fallback_posts = generate_sample_data(30)
            posts = []
            for i, post_data in enumerate(fallback_posts):
                p = Post(
                    post_id=f"{stream_id}_{i}",
                    brand=brand_name,
                    platform=platform,
                    text=post_data.get('text', ''),
                    timestamp=post_data.get('timestamp', datetime.now().isoformat()),
                    likes=post_data.get('likes', 0),
                    retweets=post_data.get('retweets', 0),
                    replies=post_data.get('replies', 0),
                    sentiment=post_data.get('sentiment', 'Neutral'),
                    polarity=post_data.get('polarity', 0.0),
                    hashtags=post_data.get('hashtags', []),
                    mentions=post_data.get('mentions', []),
                    keywords=post_data.get('keywords', []),
                    emotions=post_data.get('emotions', {}),
                    location=post_data.get('location', 'Unknown'),
                    language=post_data.get('language', 'en'),
                    user_segment=post_data.get('user_segment', 'General')
                )
                posts.append(p)
            
            db.active_streams[stream_id] = {
                'brand': brand_name,
                'platform': platform,
                'keywords': keywords,
                'data': posts,
                'status': 'active',
                'start_time': datetime.now().isoformat(),
                'alerts': []
            }
            
            return jsonify({'success': True, 'stream_id': stream_id}), 200
            
    except Exception as e:
        print(f"Error in start_realtime_monitoring: {str(e)}")
        traceback.print_exc()
        # Generate emergency fallback data
        try:
            brand_name = request.get_json().get('brand_name', 'Unknown')
            platform = request.get_json().get('platform', 'twitter')
            stream_id = f"{brand_name}_emergency_{int(datetime.now().timestamp())}"
            
            fallback_posts = generate_sample_data(20)
            posts = []
            for i, post_data in enumerate(fallback_posts):
                p = Post(
                    post_id=f"{stream_id}_{i}",
                    brand=brand_name,
                    platform=platform,
                    text=post_data.get('text', ''),
                    timestamp=datetime.now().isoformat(),
                    likes=random.randint(0, 100),
                    sentiment=post_data.get('sentiment', 'Neutral'),
                    polarity=post_data.get('polarity', 0.0)
                )
                posts.append(p)
            
            db.active_streams[stream_id] = {
                'brand': brand_name,
                'platform': platform,
                'keywords': [],
                'data': posts,
                'status': 'active',
                'start_time': datetime.now().isoformat(),
                'alerts': []
            }
            
            return jsonify({'success': True, 'stream_id': stream_id, 'fallback': True}), 200
        except:
            return jsonify({'success': False, 'error': 'Failed to start monitoring'}), 500

@app.route('/api/realtime-data/<stream_id>', methods=['GET'])
def get_realtime_data(stream_id: str):
    """Get real-time stream data from MockDB with automatic data generation."""
    try:
        print(f"Fetching real-time data for stream: {stream_id}")
        
        if stream_id not in db.active_streams:
            print(f"Stream {stream_id} not found")
            return jsonify({'success': False, 'error': 'Stream not found'}), 404
        
        stream_data = db.active_streams[stream_id]
        
        # Simulate new data arriving (70% chance)
        if random.random() < 0.7:
            new_posts_count = random.randint(1, 3)
            print(f"Generating {new_posts_count} new posts for stream {stream_id}")
            
            for _ in range(new_posts_count):
                try:
                    new_post_data = generate_sample_data(1)[0]
                    new_post = Post(
                        post_id=f"live_{stream_id}_{datetime.now().timestamp()}_{random.randint(1000, 9999)}",
                        brand=stream_data['brand'],
                        platform=stream_data['platform'],
                        text=new_post_data.get('text', ''),
                        timestamp=datetime.now().isoformat(),
                        likes=new_post_data.get('likes', 0),
                        retweets=new_post_data.get('retweets', 0),
                        replies=new_post_data.get('replies', 0),
                        sentiment=new_post_data.get('sentiment', 'Neutral'),
                        polarity=new_post_data.get('polarity', 0.0),
                        hashtags=new_post_data.get('hashtags', []),
                        mentions=new_post_data.get('mentions', []),
                        keywords=new_post_data.get('keywords', []),
                        emotions=new_post_data.get('emotions', {}),
                        location=new_post_data.get('location', 'Unknown'),
                        language=new_post_data.get('language', 'en'),
                        user_segment=new_post_data.get('user_segment', 'General')
                    )
                    db.insert_stream_post(stream_id, new_post)
                    print(f"Added new post: {new_post.post_id}")
                except Exception as post_error:
                    print(f"Error creating new post: {str(post_error)}")
                    continue
        
        current_posts = stream_data['data']
        metrics = calculate_metrics(current_posts)
        
        print(f"Returning {len(current_posts)} posts with {len(stream_data.get('alerts', []))} alerts")
        
        return jsonify({
            'success': True,
            'data': [p.__dict__ for p in current_posts],
            'metrics': metrics,
            'alerts': stream_data.get('alerts', []),
            'status': stream_data['status']
        }), 200
        
    except Exception as e:
        print(f"Error in get_realtime_data: {str(e)}")
        traceback.print_exc()
        
        # Emergency fallback - generate fresh data
        try:
            if stream_id in db.active_streams:
                stream_data = db.active_streams[stream_id]
                fallback_data = generate_sample_data(5)
                
                for i, post_data in enumerate(fallback_data):
                    p = Post(
                        post_id=f"fallback_{stream_id}_{i}_{datetime.now().timestamp()}",
                        brand=stream_data.get('brand', 'Unknown'),
                        platform=stream_data.get('platform', 'twitter'),
                        text=post_data.get('text', ''),
                        timestamp=datetime.now().isoformat(),
                        sentiment=post_data.get('sentiment', 'Neutral'),
                        polarity=post_data.get('polarity', 0.0)
                    )
                    stream_data['data'].append(p)
                
                metrics = calculate_metrics(stream_data['data'])
                
                return jsonify({
                    'success': True,
                    'data': [p.__dict__ for p in stream_data['data']],
                    'metrics': metrics,
                    'alerts': [],
                    'status': 'active',
                    'fallback': True
                }), 200
        except:
            pass
        
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop-realtime/<stream_id>', methods=['POST'])
def stop_realtime_monitoring(stream_id: str):
    """Stop real-time monitoring mock."""
    try:
        if stream_id in db.active_streams:
            db.active_streams[stream_id]['status'] = 'stopped'
            print(f"Stream {stream_id} stopped")
            return jsonify({'success': True, 'message': 'Stream stopped'}), 200
        return jsonify({'success': False, 'error': 'Stream not found'}), 404
    except Exception as e:
        print(f"Error in stop_realtime_monitoring: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/set-alerts', methods=['POST'])
def set_alerts():
    """Configure alert thresholds."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        stream_id = data.get('stream_id')
        thresholds = data.get('thresholds', {})
        
        if not stream_id:
            return jsonify({'success': False, 'error': 'Stream ID required'}), 400
        
        db.alert_thresholds[stream_id] = thresholds
        print(f"Alert thresholds set for stream {stream_id}: {thresholds}")
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"Error in set_alerts: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/regional-analysis', methods=['POST'])
def get_regional_analysis():
    """Get regional and demographic sentiment analysis."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        data_dicts = data.get('data', [])
        
        if not data_dicts:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        posts = [Post(**d) for d in data_dicts]
        regional, demographics = analyze_by_region(posts)
        
        return jsonify({
            'success': True,
            'regional': regional,
            'demographics': demographics
        }), 200
    except Exception as e:
        print(f"Error in get_regional_analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trending-analysis', methods=['POST'])
def get_trending_analysis():
    """Get trending topics and hashtag analysis."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        data_dicts = data.get('data', [])
        time_window = data.get('time_window', '24h')
        
        if not data_dicts:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        posts = [Post(**d) for d in data_dicts]
        trending = analyze_trending_topics(posts, time_window)
        
        return jsonify({
            'success': True,
            'trending': trending
        }), 200
    except Exception as e:
        print(f"Error in get_trending_analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/keyword-evolution', methods=['POST'])
def get_keyword_evolution():
    """Track keyword sentiment evolution over time."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        data_dicts = data.get('data', [])
        keyword = data.get('keyword', '')
        
        if not data_dicts or not keyword:
            return jsonify({'success': False, 'error': 'Data and keyword required'}), 400
        
        posts = [Post(**d) for d in data_dicts]
        evolution = analyze_keyword_evolution(posts, keyword)
        
        return jsonify({
            'success': True,
            'keyword': keyword,
            'evolution': evolution
        }), 200
    except Exception as e:
        print(f"Error in get_keyword_evolution: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai-summary', methods=['POST'])
def get_ai_summary():
    """Generate AI-powered insights summary."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        data_dicts = data.get('data', [])
        brand_name = data.get('brand_name', 'Brand')
        
        if not data_dicts:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        posts = [Post(**d) for d in data_dicts]
        summary = generate_ai_summary(posts, brand_name)
        
        return jsonify({
            'success': True,
            'summary': summary
        }), 200
    except Exception as e:
        print(f"Error in get_ai_summary: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/competitor-compare', methods=['POST'])
def compare_competitors():
    """Compare multiple brands side-by-side."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        brands = data.get('brands', [])
        time_period = data.get('time_period', '7d')
        
        if len(brands) < 2:
            return jsonify({'success': False, 'error': 'At least 2 brands required'}), 400
        
        main_brand = brands[0]
        competitors = brands[1:]
        
        analysis = analyze_brand_sentiment(main_brand, competitors, time_period)
        
        return jsonify({
            'success': True,
            'comparison': analysis
        }), 200
    except Exception as e:
        print(f"Error in compare_competitors: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# --- Main Execution ---

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Enhanced Brand Sentiment Analysis Platform")
    print("=" * 70)
    print("✨ Core Features:")
    print("  • Real-time sentiment monitoring with auto-fallback")
    print("  • Comprehensive error handling and recovery")
    print("  • Multi-brand competitive analysis")
    print("  • AI-powered insights and recommendations")
    print("  • Regional and demographic breakdowns")
    print("=" * 70)
    print("🌐 Server: http://localhost:5001")
    print("⚡ Press Ctrl+C to stop")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5001)