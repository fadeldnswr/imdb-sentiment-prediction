# 🎬 IMDB Movie Review Sentiment Analysis
This project implements a simple Recurrent Neural Network (RNN) architecture to perform sentiment analysis on the IMDB movie review dataset. The model classifies reviews as either positive or negative, providing a practical application of sequence modeling in Natural Language Processing (NLP).

The trained model is deployed using Streamlit, allowing users to input their own movie reviews and receive instant sentiment predictions via an interactive web interface.

## 🚀 Features
- 🔁 Simple RNN architecture for sequence modeling
- 🗂️ Trained on the IMDB movie review dataset
- 📊 Binary sentiment classification: Positive or Negative
- 🌐 Streamlit Web App for live predictions
- 📈 Real-time feedback on review sentiment

## 🧠 Model Overview
- Embedding layer to encode input words
- RNN layer for sequence learning
- Dense layer + Sigmoid activation for binary output
- Binary Cross-Entropy loss function
- Optimizer: RMSProp or Adam

## 🛠️ Tech Stack
- Python 
- TensorFlow / Keras
- Numpy & Pandas
- Streamlit