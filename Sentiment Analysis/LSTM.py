

!pip install requests beautifulsoup4 nltk transformers yfinance scikit-learn pandas matplotlib

import nltk
# Import required libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
import random
from sklearn.metrics import mean_squared_error  # Import MSE

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load pre-trained FinBERT model for sentiment analysis
nlp = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Function to analyze sentiment of news articles
def get_sentiment(text):
    cleaned_text = preprocess_text(text)
    sentiment_result = nlp(cleaned_text)
    return 1 if sentiment_result[0]['label'] == 'positive' else -1

# Function to scrape Yahoo Finance news headl

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
import random

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load pre-trained FinBERT model for sentiment analysis
nlp = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Function to analyze sentiment of news articles
def get_sentiment(text):
    cleaned_text = preprocess_text(text)
    sentiment_result = nlp(cleaned_text)
    return 1 if sentiment_result[0]['label'] == 'positive' else -1

# Function to scrape Yahoo Finance news headlines
def scrape_yahoo_finance():
    url = 'https://finance.yahoo.com/topic/stock-market-news/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the news headlines
    headlines = []
    for item in soup.find_all('h3', class_='Mb(5px)'):
        headline = item.get_text()
        headlines.append(headline)

    return headlines

# Step 1: Scrape Yahoo Finance for news articles
news_headlines = scrape_yahoo_finance()

# Step 2: Preprocess the scraped headlines
cleaned_headlines = [preprocess_text(headline) for headline in news_headlines]
print("Cleaned Headlines:", cleaned_headlines)

# Step 3: Analyze sentiment for each preprocessed headline
sentiments = [get_sentiment(headline) for headline in cleaned_headlines]
print("Sentiments:", sentiments)

# Step 4: Stock Data - Download historical stock price data for a specific company (e.g., Apple)
ticker = 'AAPL'
stock_data = yf.download(ticker, start="2023-01-01", end="2023-12-31")

# Assign random sentiments to stock data (for demonstration purposes)
stock_data['Sentiment'] = [random.choice([-1, 1]) for _ in range(len(stock_data))]

# Adding additional features: Lagged stock prices (Lag_1), Volume, Open, High, Low
stock_data['Lag_1'] = stock_data['Close'].shift(1)
stock_data.dropna(inplace=True)

# Step 5: Prepare data for LSTM prediction
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close', 'Lag_1', 'Open', 'High', 'Low', 'Volume', 'Sentiment']])

X = []
y = []

# Creating sequences for LSTM (look back of 60 days)
look_back = 60
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i])
    y.append(scaled_data[i, 0])  # Close price is the target

X, y = np.array(X), np.array(y)

# Step 6: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 8: Train the LSTM model
model.fit(X_train, y_train, batch_size=32, epochs=50)

# Step 9: Make predictions and evaluate the model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 6)))))[:, 0]

# Inverse transform the actual values for comparison
y_test_scaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 6)))))[:, 0]

# Step 10: Plot actual vs predicted stock prices
plt.figure(figsize=(10, 5))
plt.plot(y_test_scaled, label='Actual Stock Prices')
plt.plot(predictions, label='Predicted Stock Prices')
plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('Test Data Points')
plt.ylabel('Stock Price (Close)')
plt.show()

# Step 11: Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test_scaled, predictions)
print(f"Mean Squared Error: {mse}")

from sklearn.metrics import r2_score

# Calculate R-squared score
r2 = r2_score(y_test_scaled, predictions)
print(f"R-squared: {r2:.4f}")

# Define a function to calculate accuracy based on a threshold
def calculate_accuracy(y_true, y_pred, threshold=0.05):
    total = len(y_true)
    correct = 0
    for actual, predicted in zip(y_true, y_pred):
        if abs(actual - predicted) / actual <= threshold:  # Percentage difference
            correct += 1
    return correct / total * 100  # Return percentage accuracy

# Calculate accuracy with a 5% threshold
accuracy = calculate_accuracy(y_test_scaled, predictions, threshold=0.04)
print(f"Accuracy: {accuracy:.2f}%")
