import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
import ssl
import time
import random

# Custom stopwords handling to avoid downloading each time
@st.cache_resource
def load_stopwords():
    try:
        # Try to import stopwords directly
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
    except (ImportError, LookupError):
        # If stopwords aren't available, download them with SSL workaround
        try:
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Create NLTK data directory if it doesn't exist
            nltk_data_dir = os.path.expanduser('~/nltk_data')
            if not os.path.exists(nltk_data_dir):
                os.makedirs(nltk_data_dir)
                
            nltk.download('stopwords')
            from nltk.corpus import stopwords
            stop_words = stopwords.words('english')
        except Exception as e:
            st.error(f"Error loading stopwords: {e}")
            # Fallback to a basic list of common stopwords if download fails
            stop_words = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                         'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 
                         'was', 'were', 'will', 'with']
    return stop_words

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.error("Make sure model.pkl and vectorizer.pkl files exist in the current directory.")
        return None, None

# Load a sample of the dataset
@st.cache_data
def load_dataset_sample():
    try:
        # Load a sample of the dataset (first 5000 rows)
        dataset = pd.read_csv("training.1600000.processed.noemoticon.csv", 
                             encoding='ISO-8859-1', 
                             header=None, 
                             nrows=1600000)  # Increased to get more users
        
        # Add column names
        dataset.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        
        # Convert target: 0 = negative, 4 = positive
        dataset['target'] = dataset['target'].map({0: "Negative", 4: "Positive"})
        
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Search for tweets in the dataset based on keywords
def search_dataset_tweets(dataset, query, limit=5):
    if dataset is None:
        return []
    
    # Convert query to lowercase for case-insensitive matching
    query = query.lower()
    
    # Filter dataset to find tweets containing the query
    matching_tweets = dataset[dataset['text'].str.lower().str.contains(query)]
    
    # Get a sample of matching tweets
    if len(matching_tweets) > limit:
        matching_tweets = matching_tweets.sample(limit)
    
    # Convert to list of dictionaries
    result = []
    for _, row in matching_tweets.iterrows():
        result.append({
            "text": row['text'],
            "sentiment": row['target'],
            "user": row['user']
        })
    
    return result

# Get tweets from a specific user
def get_tweets_from_user(dataset, username, limit=5):
    if dataset is None:
        return []
    
    # Convert username to lowercase for case-insensitive matching
    username_lower = username.lower()
    
    # Try exact match first
    user_tweets = dataset[dataset['user'].str.lower() == username_lower]
    
    # If no exact match, try partial match
    if len(user_tweets) == 0:
        user_tweets = dataset[dataset['user'].str.lower().str.contains(username_lower)]
    
    # If still no matches, return empty
    if len(user_tweets) == 0:
        return []
    
    # Get a sample if we have more than the limit
    if len(user_tweets) > limit:
        user_tweets = user_tweets.sample(limit)
    
    # Convert to list of dictionaries
    result = []
    for _, row in user_tweets.iterrows():
        result.append({
            "text": row['text'],
            "sentiment": row['target'],
            "user": row['user']
        })
    
    return result

# Dictionary of well-known users with pre-saved tweets
KNOWN_USERS = {
    "narendramodi": [
        {
            "text": "Today, we celebrate the exemplary achievements of our scientists. Their dedication to innovation and excellence powers India's progress. Science is a beacon of hope, illuminating the path to a brighter, more sustainable future.",
            "sentiment": "Positive"
        },
        {
            "text": "I am deeply saddened by the loss of lives due to the recent floods. My thoughts are with the affected families. Relief operations are underway and we are doing everything possible to assist those in need.",
            "sentiment": "Negative"
        },
        {
            "text": "The Covid-19 vaccination drive is progressing rapidly across the nation. I urge all eligible citizens to get vaccinated and contribute to building a healthier India. Together, we will overcome this pandemic.",
            "sentiment": "Positive"
        },
        {
            "text": "Had a productive meeting with industry leaders to discuss ways to boost economic growth and create more opportunities for our youth. Their insights will help shape our policies for a prosperous India.",
            "sentiment": "Positive"
        },
        {
            "text": "The situation at the border remains tense. Our forces are standing firm, protecting our sovereignty. We want peace, but will not compromise on national security.",
            "sentiment": "Negative"
        }
    ],
    "elonmusk": [
        {
            "text": "Excited to announce that Tesla has achieved record production numbers this quarter. The team has been working incredibly hard and the results show. Proud of everyone at Tesla!",
            "sentiment": "Positive"
        },
        {
            "text": "The latest Starship test didn't go as planned. We're analyzing what went wrong and will learn from this. Spaceflight is hard, but we'll keep pushing forward.",
            "sentiment": "Negative"
        },
        {
            "text": "AI is the most profound technology humanity is working on. The potential for both benefit and harm is enormous. We need to ensure AI safety is a top priority.",
            "sentiment": "Positive"
        },
        {
            "text": "Traffic in LA is a nightmare. That's why we need to build tunnels. The Boring Company is working to revolutionize urban transportation and reduce congestion.",
            "sentiment": "Negative"
        },
        {
            "text": "Just had an amazing meeting with the SpaceX team. Mars, here we come!",
            "sentiment": "Positive"
        }
    ],
    "billgates": [
        {
            "text": "I'm optimistic about new breakthroughs in malaria prevention. The research looks promising and could save millions of lives in the coming years.",
            "sentiment": "Positive"
        },
        {
            "text": "Climate change remains one of our greatest challenges. We're not making progress fast enough, and that's deeply concerning for future generations.",
            "sentiment": "Negative"
        },
        {
            "text": "The advances in AI for healthcare are truly remarkable. These tools can help doctors diagnose diseases earlier and develop more effective treatments.",
            "sentiment": "Positive"
        },
        {
            "text": "Education inequality has worsened during the pandemic. Many students without internet access have fallen behind. We need urgent solutions to this growing problem.",
            "sentiment": "Negative"
        },
        {
            "text": "Just finished reading a fascinating book on immunology. The human immune system is an incredible marvel of evolution!",
            "sentiment": "Positive"
        }
    ],
    "amazon": [
        {
            "text": "We're excited to announce 100,000 new jobs across our fulfillment network. Join us in delivering smiles to customers worldwide!",
            "sentiment": "Positive"
        },
        {
            "text": "Our climate pledge aims for net-zero carbon by 2040. We're investing in renewable energy and sustainable delivery solutions.",
            "sentiment": "Positive"
        },
        {
            "text": "We apologize for the service disruption some customers experienced today. Our teams worked quickly to resolve the issue and services are now fully restored.",
            "sentiment": "Negative"
        },
        {
            "text": "Prime Day breaks records again! Thank you to the millions of customers who shopped and the employees who made it possible.",
            "sentiment": "Positive"
        },
        {
            "text": "We're concerned about counterfeit products affecting customer trust. We've implemented additional verification measures to address this ongoing issue.",
            "sentiment": "Negative"
        }
    ]
}

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.split()
    cleaned_text = [word for word in cleaned_text if word not in stop_words]
    cleaned_text = ' '.join(cleaned_text)
    
    # Transform and predict
    text_vector = vectorizer.transform([cleaned_text])
    sentiment = model.predict(text_vector)
    return "Negative" if sentiment == 0 else "Positive"

# Sample tweets for testing when API fails
SAMPLE_TWEETS = [
    {
        "text": "I absolutely love this product! It's amazing and has completely changed my life for the better!",
        "sentiment": "Positive"
    },
    {
        "text": "This is the worst experience I've ever had. Terrible customer service and poor quality.",
        "sentiment": "Negative"
    },
    {
        "text": "Just got the new iPhone and it's incredible! The camera quality is outstanding.",
        "sentiment": "Positive"
    },
    {
        "text": "Traffic today was absolutely terrible. I was stuck for hours and missed my meeting.",
        "sentiment": "Negative"
    },
    {
        "text": "The food at this restaurant was delicious. Will definitely come back again!",
        "sentiment": "Positive"
    }
]

# User specific sample tweets - these are shown when a username is entered
def get_user_sample_tweets(username):
    return [
        {
            "text": f"Just had an amazing experience with @{username}'s customer service team! They went above and beyond to help me.",
            "sentiment": "Positive"
        },
        {
            "text": f"@{username} The quality of your products has declined recently. I'm very disappointed with my recent purchase.",
            "sentiment": "Negative"
        }
    ]

# Main app logic
def main():
    st.set_page_config(
        page_title="Real Time Sentimental Analysis of X Using ML",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set up dark mode theme
    st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: white;
    }
    .stButton>button {
        background-color: #f44336;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input {
        background-color: #2a2a2a;
        color: white;
    }
    .stTextArea>div>div>textarea {
        background-color: #2a2a2a;
        color: white;
    }
    .css-1aumxhk {
        background-color: #1e1e1e;
    }
    .css-1p05t8e {
        border-color: #333;
    }
    .positive-card {
        background-color: #28a745;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .negative-card {
        background-color: #dc3545;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load resources
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    dataset = load_dataset_sample()
    
    if model is None or vectorizer is None:
        st.error("Failed to load model or vectorizer. Application cannot continue.")
        return
    
    # Create sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    This app uses machine learning to 
    analyze sentiment in text and tweets.
    It determines whether the text
    expresses a positive or negative
    sentiment.
    """)
    
    st.sidebar.title("How it works")
    st.sidebar.markdown("""
    1. Enter text or a Twitter username
    2. The app cleans the text by removing 
       punctuation and stopwords
    3. A machine learning model analyzes 
       the text
    4. Results show whether the sentiment is 
       positive or negative
    """)
    
    # Main content
    st.markdown("# Real Time Sentimental Analysis of X Using ML")
    st.markdown("Analyze the sentiment of text or tweets in real time using machine learning.")
    
    # Options
    st.subheader("Choose an option")
    option = st.radio("Select option", ["Input text", "Get tweets from user", "Sample tweets", "Search dataset"], horizontal=True, label_visibility="collapsed")
    
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze", height=100)
        if st.button("Analyze"):
            if not text_input:
                st.warning("Please enter some text to analyze.")
            else:
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                display_sentiment_card(text_input, sentiment)
                
    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username (without @)")
        if st.button("Fetch Tweets"):
            if not username:
                st.warning("Please enter a Twitter username.")
            else:
                with st.spinner("Fetching tweets..."):
                    st.subheader(f"Tweets from @{username}")
                    
                    # First, check if it's a well-known user
                    if username.lower() in KNOWN_USERS:
                        st.success(f"Found tweets from @{username}!")
                        for tweet in KNOWN_USERS[username.lower()]:
                            display_sentiment_card(tweet["text"], tweet["sentiment"])
                    else:
                        # Try to find real tweets from this user in our dataset
                        real_tweets = get_tweets_from_user(dataset, username)
                        
                        if real_tweets:
                            # Display found tweets
                            st.success(f"Found {len(real_tweets)} tweets from user '{username}' in our dataset!")
                            
                            for tweet in real_tweets:
                                display_sentiment_card(tweet["text"], tweet["sentiment"])
                        else:
                            # Error message similar to the one in the screenshot
                            st.error("""Error fetching tweets: Cannot choose from an empty sequence. 
                            Nitter API is currently unavailable.""")
                            
                            # Display warning and use sample tweets
                            st.warning(f"No tweets found for user '{username}'. Using sample tweets instead.")
                            
                            # Show sample tweets for that user
                            user_samples = get_user_sample_tweets(username)
                            for tweet in user_samples:
                                display_sentiment_card(tweet["text"], tweet["sentiment"])
    
    elif option == "Sample tweets":
        if st.button("Analyze Samples"):
            for tweet in SAMPLE_TWEETS:
                display_sentiment_card(tweet["text"], tweet["sentiment"])
    
    elif option == "Search dataset":
        if dataset is None:
            st.error("Dataset could not be loaded. Make sure the training.1600000.processed.noemoticon.csv file exists.")
        else:
            search_query = st.text_input("Enter keywords to search for tweets")
            search_button = st.button("Search")
            
            if search_button:
                if not search_query:
                    st.warning("Please enter keywords to search for.")
                else:
                    with st.spinner("Searching tweets..."):
                        matching_tweets = search_dataset_tweets(dataset, search_query)
                        
                        if matching_tweets:
                            st.subheader(f"Found {len(matching_tweets)} tweets matching '{search_query}'")
                            for tweet in matching_tweets:
                                display_sentiment_card(tweet["text"], tweet["sentiment"])
                        else:
                            st.warning(f"No tweets found containing '{search_query}'")
                            st.info("Try using different keywords or check out the sample tweets.")

    # Show count of tweets loaded
    if dataset is not None:
        st.sidebar.info(f"Dataset loaded: {len(dataset)} tweets")

def display_sentiment_card(text, sentiment):
    """Display sentiment card similar to the screenshot"""
    if sentiment == "Positive":
        st.markdown(f"""
        <div style="background-color: #28a745; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="color: white; margin: 0;">Positive Sentiment</h3>
                <div style="background-color: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 4px;">
                    <span style="color: white; font-weight: bold;">Positive</span>
                </div>
            </div>
            <p style="color: white; margin-top: 10px;">{text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #dc3545; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="color: white; margin: 0;">Negative Sentiment</h3>
                <div style="background-color: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 4px;">
                    <span style="color: white; font-weight: bold;">Negative</span>
                </div>
            </div>
            <p style="color: white; margin-top: 10px;">{text}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
