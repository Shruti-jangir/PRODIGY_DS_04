import pandas as pd

# Load datasets
training_path =r"C:\Users\shrut\Downloads\archive (9)\twitter_training.csv"
validation_path = r"C:\Users\shrut\Downloads\archive (9)\twitter_validation.csv"

training_data = pd.read_csv(training_path)
validation_data = pd.read_csv(validation_path)

# Inspect datasets
print("Training Data:\n", training_data.head())
print("\nValidation Data:\n", validation_data.head())


# Data Preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)  
    text = re.sub(r'[^A-Za-z\s]', '', text)  
    text = text.lower()  
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

training_data['Tweet'] = training_data['Tweet'].astype(str).fillna("")
validation_data['Tweet'] = validation_data['Tweet'].astype(str).fillna("")

training_data['Cleaned_Tweet'] = training_data['Tweet'].apply(preprocess_text)
validation_data['Cleaned_Tweet'] = validation_data['Tweet'].apply(preprocess_text)

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    return "Positive" if score['compound'] > 0.05 else "Negative" if score['compound'] < -0.05 else "Neutral"

training_data['Predicted_Sentiment'] = training_data['Cleaned_Tweet'].apply(get_sentiment)
validation_data['Predicted_Sentiment'] = validation_data['Cleaned_Tweet'].apply(get_sentiment)

# Trend Analysis
training_data['Tweet ID'] = pd.to_datetime(training_data['Tweet ID'], errors='coerce')
validation_data['Tweet ID'] = pd.to_datetime(validation_data['Tweet ID'], errors='coerce')

sentiment_trends = training_data.groupby(training_data['Tweet ID'])['Predicted_Sentiment'].value_counts().unstack().fillna(0)

sentiment_trends = sentiment_trends.astype(int)
print(sentiment_trends)

# Plot the trends
import matplotlib.pyplot as plt

sentiment_trends.plot(kind='line', figsize=(12, 6), title='Sentiment Trends by Tweet ID')
plt.xlabel('Tweet_ID')
plt.ylabel('Sentiment Count')
plt.legend(title='Sentiment')
plt.show()

#Visualizations
from wordcloud import WordCloud
#visualize training data
positive_tweets = " ".join(training_data[training_data['Predicted_Sentiment'] == "Positive"]['Cleaned_Tweet'])

wordcloud = WordCloud(background_color='white', width=800, height=400).generate(positive_tweets)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#visualize validation data

positive_tweets = " ".join(validation_data[validation_data['Predicted_Sentiment'] == "Positive"]['Cleaned_Tweet'])

wordcloud = WordCloud(background_color='white', width=800, height=400).generate(positive_tweets)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Topic Anlysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(training_data['Cleaned_Tweet'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

for i, topic in enumerate(lda.components_):
    print(f"Topic {i}: ", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])


training_data.to_csv("processed_training_data.csv", index=False)
validation_data.to_csv("processed_validation_data.csv", index=False)
