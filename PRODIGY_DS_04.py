import pandas as pd
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sample dataset
file_path = 'sample_tweets.csv'  # Ensure this is the correct path to your CSV file
data = pd.read_csv(file_path, encoding='latin1')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display the column names to understand the structure
print("\nColumn names in the dataset:")
print(data.columns)

# Rename columns for easier access if necessary (assuming standard sentiment140 dataset structure)
# If your dataset has different columns, update the column names accordingly
data.columns = ['polarity', 'id', 'date', 'query', 'user', 'tweet']


# Function to clean tweet text
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#', '', tweet)  # Remove hashtag symbol
    tweet = re.sub(r'\W', ' ', tweet)  # Remove non-word characters
    tweet = re.sub(r'\s+', ' ', tweet)  # Remove extra spaces
    tweet = tweet.strip()  # Remove leading and trailing spaces
    return tweet


# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity


# Function to analyze sentiment using VADER
def analyze_sentiment_vader(tweet):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(tweet)
    return sentiment_score['compound']


# Clean tweets
data['Cleaned_Tweet'] = data['tweet'].apply(clean_tweet)

# Analyze sentiment
data['Sentiment_TextBlob'] = data['Cleaned_Tweet'].apply(analyze_sentiment_textblob)
data['Sentiment_VADER'] = data['Cleaned_Tweet'].apply(analyze_sentiment_vader)

# Classify sentiment
data['Sentiment_TextBlob_Class'] = data['Sentiment_TextBlob'].apply(
    lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
data['Sentiment_VADER_Class'] = data['Sentiment_VADER'].apply(
    lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))


# Visualization function
def visualize_sentiments(df):
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    sns.countplot(x='Sentiment_TextBlob_Class', data=df, palette='viridis')
    plt.title('Sentiment Analysis (TextBlob)')

    plt.subplot(1, 2, 2)
    sns.countplot(x='Sentiment_VADER_Class', data=df, palette='viridis')
    plt.title('Sentiment Analysis (VADER)')

    plt.tight_layout()
    plt.show()


# Visualize sentiments
visualize_sentiments(data)
