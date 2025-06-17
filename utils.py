from tensorflow.keras.preprocessing import sequence

# Helper function to decode review
# Decoded function to convert encoded reviews back to text
def decode_review(encoded_review, reverse_word_index):
  '''
  Decodes a sequence of integers back to a human-readable string.
  Args:
    encoded_review (list of int): The encoded review as a list of integers.
    reverse_word_index (dict): A dictionary mapping integers to words.
  '''
  return " ".join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


# Preprocess function to convert reviews to a format suitable for the model
def preprocess_text(text, word_index):
  '''
  Converts a text review into a padded sequence of integers.
  Args:
    text (str): The review text to preprocess.
    word_index (dict): A dictionary mapping words to their integer indices.
  '''
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review

# Prediction function
# Prediction function
def predict_sentiment(review, model):
  '''
  Predicts the sentiment of a given review.
  Args:
    review (str): The review text to analyze.
  '''
  preprocessed = preprocess_text(review)
  prediction = model.predict(preprocessed)
  sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
  return sentiment, prediction[0][0]