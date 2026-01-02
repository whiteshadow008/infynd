"""
Social Media Sentiment Tracker - Flask Backend
AI Hackathon Minds 2025

Run with: python app.py
Access at: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from textblob import TextBlob
import tweepy
import requests
from datetime import datetime, timedelta
import re
from collections import Counter
import io
import json
import base64
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ===========================
# API CONFIGURATION
# ===========================

# Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_HEADERS = {"Authorization": "Bearer hf_2Ix4zte9YnXFYK6A97FwB6LY8"}

# Twitter API
#STWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMIL4wEAAAAAAbat2206L4xNE1CR6BzWUViDhAM%3Dvw9fE4WbV4dfbOvjesscRI6R8evnr881eqnnM9ibexPbGJDH04"
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAAwZ4wEAAAAA7ofRxe7XrEDfZ889q20dNAIi0x4%3DpE2toyN1xZkcztPGNfpuuLFMIDAlF304BMps0awsQYo8cAsSwz"
# ===========================
# SENTIMENT ANALYSIS FUNCTIONS
# ===========================

def classify_sentiment_huggingface(text):
    """Classify sentiment using Hugging Face BART model"""
    try:
        labels = ["positive", "negative", "neutral"]
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": labels}
        }
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            sentiment = result["labels"][0]
            score = result["scores"][0]
            
            if sentiment == "positive":
                polarity = score
            elif sentiment == "negative":
                polarity = -score
            else:
                polarity = 0.0
                
            return sentiment.capitalize(), polarity
        else:
            return analyze_sentiment_textblob(text)
    except:
        return analyze_sentiment_textblob(text)

def analyze_sentiment_textblob(text):
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

def extract_keywords(text):
    """Extract hashtags, mentions, and keywords"""
    hashtags = re.findall(r'#\w+', str(text))
    mentions = re.findall(r'@\w+', str(text))
    
    # Extract keywords
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'this', 'that', 'with', 'have', 'from'}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    keywords = [w for w in words if w not in stop_words]
    
    return hashtags, mentions, keywords

# ===========================
# DATA GENERATION & PROCESSING
# ===========================

def generate_sample_data(num_posts=100):
    """Generate sample social media posts"""
    sample_posts = [
        "This product is amazing! Best purchase ever! #love #awesome",
        "Terrible customer service. Very disappointed. #fail",
        "It's okay, nothing special but does the job.",
        "Love the new features! Keep up the great work! @brandname",
        "Worst experience. Will never buy again. #disappointed",
        "Good quality for the price. Would recommend.",
        "Absolutely fantastic! Exceeded expectations! #winning",
        "Not what I expected. Pretty disappointing.",
        "Decent product, would recommend to friends.",
        "Outstanding service! Very happy with my purchase! #happy",
        "The quality is poor. Not worth the money.",
        "Amazing experience! Will definitely come back! #bestever",
        "Mediocre at best. Expected more for the price.",
        "Excellent product! Five stars! @company #excellent",
        "Horrible. Complete waste of money. #regret",
        "Pretty good overall. A few minor issues.",
        "Love it! Exactly what I was looking for! #perfect",
        "Not impressed. Customer support was unhelpful.",
        "Great value for money! #deals #shopping",
        "Disappointed with the quality. Expected better."
    ]
    
    platforms = ['Twitter', 'Facebook', 'Instagram', 'Reddit', 'LinkedIn']
    
    data = []
    for i in range(num_posts):
        post = sample_posts[i % len(sample_posts)]
        timestamp = datetime.now() - timedelta(hours=i, minutes=i*5)
        
        sentiment, polarity = analyze_sentiment_textblob(post)
        hashtags, mentions, keywords = extract_keywords(post)
        
        data.append({
            'post_id': i + 1,
            'text': post,
            'platform': platforms[i % len(platforms)],
            'author': f'@user{i+1}',
            'timestamp': timestamp.isoformat(),
            'likes': (i * 7) % 100,
            'retweets': (i * 3) % 50,
            'replies': (i * 2) % 30,
            'sentiment': sentiment,
            'polarity': polarity,
            'hashtags': hashtags,
            'mentions': mentions,
            'keywords': keywords
        })
    
    return data

def fetch_twitter_data(username, num_tweets, use_huggingface=False):
    """Fetch tweets from Twitter API"""
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        
        # Get user data
        user_data = client.get_user(username=username)
        
        if not user_data.data:
            return {"error": f"No user found with username @{username}"}
        
        # Fetch tweets
        tweets = client.get_users_tweets(
            id=user_data.data.id,
            max_results=min(num_tweets, 100),
            tweet_fields=["created_at", "public_metrics", "text"],
            exclude=["retweets", "replies"]
        )
        
        if not tweets.data:
            return {"error": f"No tweets found for @{username}"}
        
        # Process tweets
        data = []
        for tweet in tweets.data:
            text = tweet.text
            
            # Sentiment analysis
            if use_huggingface:
                sentiment, polarity = classify_sentiment_huggingface(text)
            else:
                sentiment, polarity = analyze_sentiment_textblob(text)
            
            # Extract features
            hashtags, mentions, keywords = extract_keywords(text)
            
            # Get metrics
            metrics = tweet.public_metrics
            
            data.append({
                'post_id': tweet.id,
                'text': text,
                'sentiment': sentiment,
                'polarity': polarity,
                'platform': 'Twitter',
                'author': f'@{username}',
                'timestamp': tweet.created_at.isoformat(),
                'likes': metrics.get('like_count', 0),
                'retweets': metrics.get('retweet_count', 0),
                'replies': metrics.get('reply_count', 0),
                'hashtags': hashtags,
                'mentions': mentions,
                'keywords': keywords
            })
        
        return data
        
    except Exception as e:
        return {"error": str(e)}

def process_csv_data(file_content, use_huggingface=False):
    """Process uploaded CSV file"""
    try:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        
        if 'text' not in df.columns:
            return {"error": "CSV must contain a 'text' column"}
        
        # Add missing columns
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now().isoformat()
        if 'platform' not in df.columns:
            df['platform'] = 'Unknown'
        if 'author' not in df.columns:
            df['author'] = 'Anonymous'
        if 'likes' not in df.columns:
            df['likes'] = 0
        if 'retweets' not in df.columns:
            df['retweets'] = 0
        
        # Analyze sentiment
        data = []
        for idx, row in df.iterrows():
            if use_huggingface:
                sentiment, polarity = classify_sentiment_huggingface(row['text'])
            else:
                sentiment, polarity = analyze_sentiment_textblob(row['text'])
            
            hashtags, mentions, keywords = extract_keywords(row['text'])
            
            data.append({
                'post_id': idx + 1,
                'text': row['text'],
                'platform': row.get('platform', 'Unknown'),
                'author': row.get('author', 'Anonymous'),
                'timestamp': row.get('timestamp', datetime.now().isoformat()),
                'likes': int(row.get('likes', 0)),
                'retweets': int(row.get('retweets', 0)),
                'replies': int(row.get('replies', 0)),
                'sentiment': sentiment,
                'polarity': polarity,
                'hashtags': hashtags,
                'mentions': mentions,
                'keywords': keywords
            })
        
        return data
        
    except Exception as e:
        return {"error": str(e)}

# ===========================
# ANALYTICS FUNCTIONS
# ===========================

def calculate_metrics(data):
    """Calculate sentiment metrics"""
    if not data or 'error' in data:
        return {}
    
    df = pd.DataFrame(data)
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

def get_trending_keywords(data, limit=10):
    """Get trending hashtags and keywords"""
    if not data or 'error' in data:
        return {'hashtags': [], 'mentions': [], 'keywords': []}
    
    df = pd.DataFrame(data)
    
    # Hashtags
    all_hashtags = []
    for tags in df['hashtags']:
        all_hashtags.extend(tags)
    hashtag_counts = Counter(all_hashtags).most_common(limit)
    
    # Mentions
    all_mentions = []
    for mentions in df['mentions']:
        all_mentions.extend(mentions)
    mention_counts = Counter(all_mentions).most_common(limit)
    
    # Keywords
    all_keywords = []
    for keywords in df['keywords']:
        all_keywords.extend(keywords)
    keyword_counts = Counter(all_keywords).most_common(limit)
    
    return {
        'hashtags': [{'tag': tag, 'count': count} for tag, count in hashtag_counts],
        'mentions': [{'mention': mention, 'count': count} for mention, count in mention_counts],
        'keywords': [{'keyword': keyword, 'count': count} for keyword, count in keyword_counts]
    }

def generate_wordcloud_image(data, sentiment_filter=None):
    """Generate word cloud image"""
    try:
        df = pd.DataFrame(data)
        
        if sentiment_filter:
            df = df[df['sentiment'] == sentiment_filter]
        
        text = ' '.join(df['text'].tolist()).lower()
        text = re.sub(r'http\S+|[^a-z\s]', '', text)
        
        if not text.strip():
            return None
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis'
        ).generate(text)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        return None

def generate_insights(data):
    """Generate AI insights summary"""
    if not data or 'error' in data:
        return "No data available for analysis"
    
    metrics = calculate_metrics(data)
    df = pd.DataFrame(data)
    
    # Platform analysis
    platform_counts = df['platform'].value_counts()
    most_active_platform = platform_counts.index[0] if len(platform_counts) > 0 else 'N/A'
    
    # Time analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    peak_hour = df['hour'].mode()[0] if len(df) > 0 else 'N/A'
    
    # Trend
    if len(df) > 1:
        recent_data = df.nlargest(int(len(df) * 0.3), 'timestamp')
        recent_negative_pct = (len(recent_data[recent_data['sentiment'] == 'Negative']) / len(recent_data)) * 100
        trend = "increasing" if recent_negative_pct > metrics['negative_pct'] else "decreasing"
    else:
        trend = "stable"
    
    summary = f"""
📊 Executive Summary

Total Mentions: {metrics['total_posts']}
Overall Sentiment: {metrics['positive_pct']}% Positive | {metrics['neutral_pct']}% Neutral | {metrics['negative_pct']}% Negative

🔍 Key Findings

Most active platform: {most_active_platform}

Peak activity hour: {peak_hour}:00

Negative sentiment trend: {trend}

Total engagement: {metrics['total_engagement']} interactions

Recommendations:"""
    if metrics['negative_pct'] > 30:
        summary += "\n- 🚨 HIGH PRIORITY: Address negative sentiment immediately\n- Respond to complaints within 24 hours"
    elif metrics['negative_pct'] > 15:
        summary += "\n- ⚠️ MODERATE: Monitor negative feedback closely\n- Improve customer satisfaction"
    else:
        summary += "\n- ✅ HEALTHY: Continue positive engagement strategies"
    
    if metrics['positive_pct'] > 50:
        summary += "\n- 🌟 Leverage positive sentiment in marketing\n- Encourage testimonials"
    
    return summary

# ===========================
# FLASK ROUTES
# ===========================

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample data"""
    num_posts = int(request.args.get('num_posts', 100))
    data = generate_sample_data(num_posts)
    metrics = calculate_metrics(data)
    trending = get_trending_keywords(data)
    
    return jsonify({
        'success': True,
        'data': data,
        'metrics': metrics,
        'trending': trending
    })

@app.route('/api/twitter-data', methods=['POST'])
def get_twitter_data():
    """Fetch Twitter data"""
    username = request.json.get('username')
    num_tweets = int(request.json.get('num_tweets', 50))
    use_huggingface = request.json.get('use_huggingface', False)
    
    if not username:
        return jsonify({'success': False, 'error': 'Username is required'})
    
    data = fetch_twitter_data(username, num_tweets, use_huggingface)
    
    if isinstance(data, dict) and 'error' in data:
        return jsonify({'success': False, 'error': data['error']})
    
    metrics = calculate_metrics(data)
    trending = get_trending_keywords(data)
    
    return jsonify({
        'success': True,
        'data': data,
        'metrics': metrics,
        'trending': trending
    })

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Upload and process CSV"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    use_huggingface = request.form.get('use_huggingface', 'false') == 'true'
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'error': 'File must be CSV'})
    
    data = process_csv_data(file.read(), use_huggingface)
    
    if isinstance(data, dict) and 'error' in data:
        return jsonify({'success': False, 'error': data['error']})
    
    metrics = calculate_metrics(data)
    trending = get_trending_keywords(data)
    
    return jsonify({
        'success': True,
        'data': data,
        'metrics': metrics,
        'trending': trending
    })

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze single text"""
    text = request.json.get('text')
    platform = request.json.get('platform', 'Manual')
    author = request.json.get('author', '@user')
    use_huggingface = request.json.get('use_huggingface', False)
    
    if not text:
        return jsonify({'success': False, 'error': 'Text is required'})
    
    if use_huggingface:
        sentiment, polarity = classify_sentiment_huggingface(text)
    else:
        sentiment, polarity = analyze_sentiment_textblob(text)
    
    hashtags, mentions, keywords = extract_keywords(text)
    
    post = {
        'post_id': int(datetime.now().timestamp()),
        'text': text,
        'platform': platform,
        'author': author,
        'timestamp': datetime.now().isoformat(),
        'likes': 0,
        'retweets': 0,
        'replies': 0,
        'sentiment': sentiment,
        'polarity': polarity,
        'hashtags': hashtags,
        'mentions': mentions,
        'keywords': keywords
    }
    
    return jsonify({
        'success': True,
        'post': post
    })

@app.route('/api/wordcloud', methods=['POST'])
def get_wordcloud():
    """Generate word cloud"""
    data = request.json.get('data', [])
    sentiment_filter = request.json.get('sentiment_filter')
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'})
    
    img = generate_wordcloud_image(data, sentiment_filter)
    
    if img is None:
        return jsonify({'success': False, 'error': 'Could not generate word cloud'})
    
    return jsonify({
        'success': True,
        'image': img
    })

@app.route('/api/insights', methods=['POST'])
def get_insights():
    """Generate insights"""
    data = request.json.get('data', [])
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'})
    
    insights = generate_insights(data)
    
    return jsonify({
        'success': True,
        'insights': insights
    })

@app.route('/api/export', methods=['POST'])
def export_data():
    """Export data as CSV"""
    data = request.json.get('data', [])
    
    if not data:
        return jsonify({'success': False, 'error': 'No data to export'})
    
    df = pd.DataFrame(data)
    
    # Convert lists to strings
    for col in ['hashtags', 'mentions', 'keywords']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    # Create CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return jsonify({
        'success': True,
        'csv': output.getvalue()
    })

# ===========================
# RUN APPLICATION
# ===========================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Social Media Sentiment Tracker - Flask Backend")
    print("=" * 60)
    print("Server starting at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)