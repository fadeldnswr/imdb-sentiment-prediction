# Import necessary libraries
import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

from utils import decode_review, preprocess_text, predict_sentiment

# Load the IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load the pre-trained model
model = load_model("model/rnn_imdb.h5")

# Streamlit Application
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")

# Input text area for the review
user_input = st.text_area("Review Text")
if st.button("Predict"):
  # Preprocess the input text
  preprocess_input = preprocess_text(user_input, word_index)
  # Create prediction
  prediction = model.predict(preprocess_input)
  # Calculate sentiment
  sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
  # Display the result
  st.write(f"Sentiment: {sentiment}")
  st.write(f"Prediction Score: {prediction[0][0]:.4f}")
else:
  st.write("Please enter a review and click 'Predict' to see the sentiment analysis result.")